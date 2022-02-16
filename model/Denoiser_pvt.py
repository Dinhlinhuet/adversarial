import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from functools import partial

from model.UNet_Denoiser import UNet
from model.SegNet_Denoiser import SegNet
from model.AgNet.core.AgNet_Denoiser import AG_Net
from loss import onehot2norm, dice_loss
from model.pvt_v2 import PyramidVisionTransformerEncoder
from model.pvt_v2_decoder import PyramidVisionTransformerDecoder, Block
from model.DenoiseModules import BasicUformerLayer


# from pytorch_msssim import SSIM,ssim
# ssim_module = SSIM(data_range=255, size_average=True, channel=3)
def get_net(input_size, channels, model_path):
    # input_size = [299, 299]
    input_size = input_size
    denoiser = Denoise(input_size, in_chans=channels, out_chans=channels)

    # if os.path.exists(model_path):
    if model_path:
        print('load defense model', model_path)
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

class Denoise(nn.Module):
    def __init__(self, img_size=256, in_chans=3, out_chans=3,
                 embed_dim=[64, 160, 256, 512], depths=[2, 2, 2, 2], num_heads=[1, 2,4, 8],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, se_layer=False,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 **kwargs):
        super().__init__()
        self.num_mid = 4
        # self.encoder = PyramidVisionTransformerEncoder(img_size=img_size, in_chans=in_chans, patch_size=4,
        #                                                embed_dims=[32, 64, 160, 256],
        #                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        #                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
        #                                                sr_ratios=[8, 4, 2, 1], )
        # self.decoder = PyramidVisionTransformerDecoder(img_size=img_size, patch_size=4, out_chans= out_chans,
        #                                                embed_dims=[256, 160, 64, 32],
        #                                                num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 8, 8], qkv_bias=True,
        #                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 4, 3],
        #                                                sr_ratios=[1, 2, 4, 8], )
        # self.encoder = PyramidVisionTransformerEncoder(img_size=img_size, in_chans=in_chans,
        #                                                embed_dims=[64, 160, 256, 512],
        #                                                num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        #                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
        #                                                sr_ratios=[8, 4, 2, 1], )
        # self.decoder = PyramidVisionTransformerDecoder(img_size=img_size, patch_size=4, out_chans= out_chans,
        #                                                embed_dims=[512, 256, 160, 64],
        #                                                num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 8, 8], qkv_bias=True,
        #                                                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 4, 3],
        #                                                sr_ratios=[1, 2, 4, 8], num_mid=self.num_mid)
        #concate
        self.encoder = PyramidVisionTransformerEncoder(img_size=img_size, in_chans=in_chans,
                                                       embed_dims=[64, 160, 256, 512],
                                                       num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                                                       sr_ratios=[8, 4, 2, 1], )
        self.bottle_neck = Block(
                dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True,
                drop=0., attn_drop=attn_drop_rate, drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=4, linear=False)
        self.decoder = PyramidVisionTransformerDecoder(img_size=img_size, out_chans= out_chans,
                                                       embed_dims=[1024, 512, 320, 128],
                                                       out_dims=[256, 160, 64, out_chans],
                                                       num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 8, 8], qkv_bias=True,
                                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 4, 3],
                                                       sr_ratios=[1, 2, 4, 8], num_mid=self.num_mid)
        mid_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.middle_blocks = nn.ModuleList([
            BasicUformerLayer(dim=embed_dim[i],
                            output_dim=embed_dim[i]//2,
                            input_resolution=(img_size // 2**i,
                                                img_size // 2**i),
                            depth=depths[i],
                            num_heads=num_heads[i],
                            win_size=win_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=mid_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(self.num_mid)])

    def forward(self, x):
        out, feature_maps = self.encoder(x)
        # print("len", len(self.fwd))
        # print('enc', out.shape)
        out_mids = []
        for i in range(self.num_mid):
            # print('feat', feature_maps[i].shape)
            out_mid = self.middle_blocks[i](feature_maps[i])
            # print('mid', out_mid.shape)
            out_mids.append(out_mid)
        # print("out", out.shape)
        H,W = [int(np.sqrt(out.shape[1]))]*2
        out = self.bottle_neck(out, H,W)
        # print('out bottle', out.shape)
        out = self.decoder(out, out_mids)
        # print('input', torch.min(x), torch.max(x))
        # out = self.sigmoid(out)
        # print("out", torch.min(out), torch.max(out), out.shape)
        out = x+out
        # out = x - out
        return out

class Net(nn.Module):
    def __init__(self, input_size, channels, num_class, n, hard_mining=0, loss_norm=False):
        super(Net, self).__init__()
        # print('inut', input_size)
        self.denoise = Denoise(img_size=input_size)
        self.net = UNet(self.denoise, channels, num_class)
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
            adv_outputs = F.softmax(adv_outputs_, dim=1)
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
