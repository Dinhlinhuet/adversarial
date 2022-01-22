import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.UNet_Denoiser import UNet
from model.SegNet_Denoiser import SegNet
from model.AgNet.core.AgNet_Denoiser import AG_Net
from loss import onehot2norm, dice_loss
from torch.autograd import Variable
from model.transf import Transformer, _weights_init, Attention, BasicBlock
from einops import rearrange
import torch.nn.init as init
# from pytorch_msssim import SSIM,ssim
# ssim_module = SSIM(data_range=255, size_average=True, channel=3)
def get_net(height,width,num_class, channels, model_path, batch_size):
    # input_size = [299, 299]
    input_size = [256,256]
    block = Conv
    fwd_out = [64, 128, 256, 256, 256]
    num_fwd = [2, 3, 3, 3, 3]
    back_out = [64, 128, 256, 256]
    num_back = [2, 3, 3, 3]
    n = 2
    hard_mining = 0
    loss_norm = False
    # print('channel', channels)
    # return this orig_outputs[-1], adv_outputs[-1], loss
    # net = Net(input_size,channels, num_class, block, fwd_out, num_fwd, back_out, num_back, n, hard_mining, loss_norm)
    # loss = Loss(n, hard_mining, loss_norm)
    denoiser = Denoise(input_size[0], input_size[1], block, channels, fwd_out, num_fwd, back_out, num_back,
                        BasicBlock, num_blocks=[3, 3, 3], dim=256, num_tokens=8, mlp_dim=32, heads=8, depth=6,
                       emb_dropout=0.1, dropout=0.1, batch_size=batch_size
                       )

    # if os.path.exists(model_path):
    if model_path:
        print('load target model', model_path)
        pretrain_dict = torch.load(model_path)
        state_dict = denoiser.state_dict()
        # state_dict = net.denoise.state_dict()
        # print('ptr',pretrain_dict.keys())
        for key in pretrain_dict.keys():
            # assert state_dict.has_key(key)
            # print('key', key)
            assert key in state_dict
            # if key not in state_dict:
            #     # print('key', key)
            #     continue
            value = pretrain_dict[key]
            if not isinstance(value, torch.FloatTensor):
                value = value.data
            state_dict[key] = value
        denoiser.load_state_dict(state_dict)
    return denoiser

class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert (hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        # loss = torch.pow(torch.abs(x - y), self.n) / self.n
        loss = torch.pow(torch.abs(x - y), self.n)
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]

        loss = loss.mean()
        # loss = loss.sum()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss

class Loss(nn.Module):
    def __init__(self, n, hard_mining=0, norm=False):
        super(Loss, self).__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)

    def forward(self, x, y):
        z = []
        # print('lenx', len(x))
        for i in range(len(x)):
            z.append(self.loss(x[i], y[i]))
        return z

# DUNET
class Denoise(nn.Module):
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back, tfblock,
                  num_blocks, dim=256, num_tokens=8, mlp_dim=256, heads=8, depth=6,
                 emb_dropout=0.1, dropout=0.1, batch_size = 4
                 ):
        # super(Denoise, self).__init__()
        super().__init__()
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(tfblock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(tfblock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(tfblock, 64, num_blocks[2], stride=2)  # 8x8 feature maps (64 in total)
        self.apply(_weights_init)

        self.tranf_channels = 256
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(batch_size, self.L, self.tranf_channels),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(batch_size, self.tranf_channels, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  # initialized based on the paper

        # self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.q_project =  nn.Parameter(torch.empty(1, self.tranf_channels, self.tranf_channels),requires_grad=True)
        self.k_project = nn.Parameter(torch.empty(1, self.tranf_channels, self.tranf_channels),requires_grad=True)
        torch.nn.init.normal_(self.q_project, std=.02)
        torch.nn.init.normal_(self.k_project, std=.02)
        self.to_cls_token = nn.Identity()
        self.softmax = nn.Softmax()
        self.unflatten1 = nn.Unflatten(1, (32,32))

        # self.nn1 = nn.Linear(dim,
        #                      num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        if block is Bottleneck:
            expansion = 4
        else:
            expansion = 1
        # print('fwdin', fwd_in)
        # print('fwdout', fwd_out)
        # print('type', block)
        fwd = []
        n_in = fwd_in
        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    # print('d',fwd_out[i])
                    # print('nin', n_in)
                    group.append(block(n_in, fwd_out[i], stride=stride))
                    # group.append(block(1, 2, stride=stride))
                else:
                    group.append(block(fwd_out[i] * expansion, fwd_out[i]))
            n_in = fwd_out[i] * expansion
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        upsample = []
        back = []
        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in range(len(num_back) - 1, -1, -1):
            upsample.insert(0, nn.Upsample(size=(h[i], w[i]), mode='bilinear'))
            group = []
            for j in range(num_back[i]):
                if j == 0:
                    group.append(block(n_in, back_out[i]))
                else:
                    group.append(block(back_out[i] * expansion, back_out[i]))
            if i != 0:
                n_in = (back_out[i] + fwd_out[i - 1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back = nn.ModuleList(back)

        self.final = nn.Conv2d(back_out[0] * expansion, fwd_in, kernel_size=1, bias=False)
        n = 2
        # print('L{} loss'.format(n))
        hard_mining = 0
        loss_norm = False
        self.loss = Loss(n, hard_mining, loss_norm)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # out = svd_rgb(x,150,150,150)
        out = x
        outputs = []
        # print("len", len(self.fwd))
        for i in range(len(self.fwd)):
            # print('inp', out.size())
            out = self.fwd[i](out)
            # print('i', i)
            if i==len(self.fwd)-2:
                # print('in', out.size())
                out = rearrange(out,
                              'b c h w -> b (h w) c')  # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP
                out1 = out.clone()
                out2 = out.clone()
                # Tokenization
                wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
                # print('wa', out.size(), wa.size())
                A = torch.einsum('bij,bjk->bik', out1, wa)
                A = rearrange(A, 'b h w -> b w h')  # Transpose
                A = A.softmax(dim=-1)

                VV = torch.einsum('bij,bjk->bik', out2, self.token_wV)
                T = torch.einsum('bij,bjk->bik', A, VV)
                # print('T',T.size())

                cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                feat_in = torch.cat((cls_tokens, T), dim=1)
                # print('bef po', feat_in.size())
                feat_in += self.pos_embedding
                feat_in = self.dropout(feat_in)
                # print('after dropout', feat_in.size())
                feat_in = self.transformer(feat_in)  # main game
                # print('out', out.size(), self.q_project.size(), self.k_project.size(), cls_tokens.size())
                # print('pos embed', self.pos_embedding.size(), out.size(), self.q_project.size(), feat_in.size())
                XW = torch.einsum('xij,bjk->bik', out, self.q_project)
                # print('XW', XW.size())
                TW = torch.einsum('xij,bjk->bik', feat_in, self.k_project)
                XT = torch.einsum('bij,bjk->bik', XW, rearrange(TW, 'b l c -> b c l'))
                prod = torch.einsum('xij,bjk->bik',self.softmax(XT), feat_in)
                out += prod
                out = self.unflatten1(out)
                out = rearrange(out,'b h w c -> b c h w')
                # print('out trans', out.size())
                print('out tra', out.min(), out.max())
            if i != len(self.fwd) - 1:
                outputs.append(out)

        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        # print('in1', out.size())
        # out = self.attn2(out)[0]
        out = self.final(out)
        # print('input', torch.min(x), torch.max(x))
        out += x
        # out = torch.clamp(out, 0, 1)
        # norm = torch.sqrt(torch.sum(out**2))
        # out = out / norm
        out = self.sigmoid(out)
        # print("out", torch.min(out), torch.max(out))
        return out


class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# basic C
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv3 = nn.Conv2d(n_out, n_out * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_out * expansion)

        self.downsample = None
        if stride != 1 or n_in != n_out * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_in, n_out * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_out * expansion))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Net(nn.Module):
    def __init__(self, input_size, channels, num_class, block, fwd_out, num_fwd, back_out,
                 num_back, n, num_blocks, hard_mining=0, loss_norm=False, num_classes=10,
                 dim=128, num_tokens=8, mlp_dim=256, heads=8, depth=6,
                 emb_dropout=0.1, dropout=0.1, batch_size = 4):
        super(Net, self).__init__()
        # print('inut', input_size)
        self.denoise = Denoise(input_size[0], input_size[1], block, channels, fwd_out, num_fwd, back_out, num_back,
                               batch_size, num_blocks, num_classes=10, dim=128, num_tokens=8, mlp_dim=256, heads=8, depth=6,
                 emb_dropout=0.1, dropout=0.1, batch_size = 4)
        # self.net = Inception3(denoise, channels)
        self.net = UNet(self.denoise, channels,num_class)
        # self.net = SegNet(denoise, channels, num_class)
        # self.net = AG_Net(denoise, num_class)
        self.loss = Loss(n, hard_mining, loss_norm)

    def forward(self, device, orig_x, adv_x=None, train=True, defense=False):
        # print('defen', defense)
        # denoised_imgs, orig_outputs = self.net(orig_x,defense=False)
        if defense:
            # denoised_imgs, orig_outputs_ = self.net(orig_x, defense)
            denoised_imgs, orig_outputs_ = self.net(orig_x, defense)
        else:
            orig_outputs_ = self.net(orig_x, defense)
        orig_outputs = F.softmax(orig_outputs_, dim=1)
        # orig_outputs = onehot2norm(orig_outputs, device)
        # if requires_control:
        #     control_outputs = self.net(adv_x)
        #     control_loss = self.loss(control_outputs, orig_outputs)

        # if train:
        #     adv_x.volatile = False
        #     for i in range(len(orig_outputs)):
        #         orig_outputs[i].volatile = False
        # print('ori', torch.min(orig_outputs), torch.max(orig_outputs))
        if train:
            adv_denoised, adv_outputs_ = self.net(adv_x, defense=True)
            adv_outputs = F.softmax(adv_outputs_, dim = 1)
            # adv_outputs = onehot2norm(adv_outputs, device)
            # print('adv', adv_outputs.size(), torch.min(adv_outputs), torch.max(adv_outputs))
            # adv_outputs = self.net(adv_x, defense=False)
            # adv_denoised = svd_rgb(adv_denoised, 150, 150, 150)
            # ssim_module = SSIM(data_range=255, size_average=True, channel=3)
            # ssim_loss = ssim_module(orig_x,adv_x)
            # ssim_loss = ssim(orig_x, adv_denoised, data_range=1, size_average=True)
            # print('ssd', ssim_loss)
            loss = self.loss(adv_outputs, orig_outputs)
            # loss = dice_loss(orig_outputs_, adv_outputs)
            # loss.requires_grad=True
            # loss = Variable(loss.data, requires_grad=True)
            # print('orig_outputs', orig_outputs[0].size())
            # if not requires_control:
            return orig_outputs_, adv_outputs_, loss
        else:
            if defense:
                return orig_outputs_, denoised_imgs
            else:
                return orig_outputs_
            # return orig_outputs
        # else:
        #     return orig_outputs[-1], adv_outputs[-1], loss, control_outputs[-1], control_loss
        # if not requires_control:
        #     return orig_outputs[0], adv_outputs[0], loss
        # else:
        #     return orig_outputs[0], adv_outputs[0], loss, control_outputs[0], control_loss

