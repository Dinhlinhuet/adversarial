import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import random
import sys
import os
import cv2
from torch.autograd import Variable
import math
from torchvision.utils import save_image

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))

from dataset.dataset import SampleDataset, AgDataset
from pytorch_msssim import ms_ssim, ssim,  SSIM, MS_SSIM
from util import make_one_hot
from dag_utils import generate_target, generate_target_swap, generate_target_bs, generate_target_swap_cls
sys.path.append('../')
from model.stargan.model import Generator
from model.AgNet.core.models import AG_Net
from model import deeplab
from model import UNet, SegNet, DenseNet
from attacks import semantic_adv
from attacks.semantic import semantic_attack
from opts import get_args


def load_data(args):
    
    data_path = args.data_path
    n_classes = args.classes

    # generate loader
    # test_dataset = AgDataset(data_path, n_classes, args.channels, 'adv',args.model,args.attacks, None,'org',
    #                              args.width, args.height)
    test_dataset = AgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                 args.attacks, args.target, 'org', args.width, args.height, args.mask_type, suffix=args.suffix)
    # test_dataset = SampleDataset(data_path, n_classes, args.channels, 'adv', args.mode, args.model, None,\
    #                              'org', args.width, args.height, args.mask_type)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    
    return test_dataset, test_loader

def Attack(generator, model, test_loader,device, args):
    num_iterations= 500

    batch_size = args.batch_size
    generator.eval()
    # for param in generator.parameters():
    #     param.requires_grad = False
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    target_class = args.target
    adversarial_examples = []
    # adv_dir = './output/adv/{}/train/{}/{}/{}/'.format(args.data_path,args.model,args.attacks, target_class)
    # adv_dir = './output/adv/{}/val/{}/{}/{}/'.format(args.data_path, args.model, args.attacks, target_class)
    # if 'oct' in args.data_path:
    #     adv_dir = './output/adv/{}/{}/{}/{}/{}/'.format(args.data_path, args.mode, args.model, args.attacks,target_class)
    # else:
    adv_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(args.data_path, args.mode, args.model, args.attacks, args.mask_type,
                                                        target_class)
    # adv_dir = './output/adv/{}/{}/{}/{}/m{}t{}_tv/'.format(args.data_path, args.mode, args.model, args.attacks, args.mask_type,
    #                                                     target_class)
    # adv_dir = './output/adv/{}/{}/{}/{}/org_cond/'.format(args.data_path, args.mode, args.model, args.attacks, args.mask_type,
    #                                                     target_class)
    # adv_dir = './output/adv/{}/{}/{}/{}/pure_trg/'.format(args.data_path, args.mode, args.model, args.attacks, args.mask_type,
    #                                                     target_class)
    # adv_dir = './output/adv/{}/{}/{}/{}/mix_label/'.format(args.data_path, args.mode, args.model, args.attacks, args.mask_type,
    #                                                     target_class)
    print('adv dir', adv_dir)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    print('data path', args.data_path)
    if args.data_path =='fundus':
        if args.mode=='test':
            target_path = './data/fundus/test/gt/T0102.jpg'
        else:
            # target_path = './data/fundus/train/gt/g0024.jpg'
            target_path = './data/fundus/test/gt/T0102.jpg'
        adv_target = cv2.imread(target_path,0)
        adv_target[adv_target < 100] = 0
        adv_target[(adv_target >= 100) & (adv_target <= 150)] = 1
        adv_target[adv_target > 150] = 2
    else:
        if args.mode=='train':
            target_path = './data/brain/train/gt/5.png'
        else:
            target_path = './data/brain/test/gt/13.png'
        adv_target = cv2.imread(target_path, 0)
        adv_target[adv_target < 100] = 0
        adv_target[adv_target > 150] = 1
    adv_target0 = torch.tensor(adv_target).unsqueeze(0)
    print('add', adv_target0.shape, torch.unique(adv_target0))

    adversary = semantic_attack.FP_CW(0.01, num_iterations, early_stop=True, device=device)
    # adversary = semantic_attack.FP_CW_TV(0.05, num_iterations, device=device)
    criterian = nn.MSELoss()
    targeted = True
    org_imgs, adv_imgs = [],[]
    for batch_idx, (image, label) in enumerate(test_loader):
        # print('image', image.size())
        # print('label', label.size())
        # if batch_idx<280: continue
        bcz= image.shape[0]
        label = label.float()
        org_label = label.unsqueeze(1).clone().to(device)
        # print('org', org_label.shape)
        # print('label',label.shape)
        if args.model == 'AgNet' and args.data_path=='brain':
            image = image.double()
        else:
            image = image.float()
        org_imgs.append(image)
        # image = image.unsqueeze(0)
        # print('un', np.unique(pure_label))
        # image , label = image.clone().detach().requires_grad_(True), label.clone().detach().float()
        # print('lable', label.max())
        image, label = image.to(device), label.to(device)
        label_oh = make_one_hot(label, n_classes, device)
        print(args.attacks)
        target_class = int(target_class)
        if args.mask_type == '1':
            print('dagA')
            adv_target = torch.zeros_like(label)
            # adv_target = make_one_hot(adv_target, n_classes, device)
            print("target ", adv_target.shape, label.shape)

        elif args.mask_type == '2':
            print('dagB')
            adv_target,swapped = generate_target_swap(label_oh.cpu().numpy())
            adv_target=torch.from_numpy(adv_target).float().to(device)
            # adv_target=make_one_hot(adv_target, n_classes, device)

        elif args.mask_type == '3':
            print('dagC')
            # choice one randome particular class except background class(0)
            # unique_label = torch.unique(label)
            # target_class = int(random.choice(unique_label[1:]).item())
            # target_class = 2
            adv_target = generate_target_bs(batch_idx, label, target_class=target_class)
            # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
            # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
            # adv_target=make_one_hot(adv_target, n_classes, device)
        elif args.mask_type == '4':
            print('dagD')
            # choice one randome particular class except background class(0)
            # unique_label = torch.unique(label)
            # target_class = int(random.choice(unique_label[1:]).item())
            # target_class = 2
            adv_target = generate_target_swap_cls(batch_idx, label, target_class=target_class)
            # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
            # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
            # adv_target = make_one_hot(adv_target, n_classes, device)
        elif args.mask_type == '5':
            print('new')
            adv_target = adv_target0.repeat(bcz,1,1).float()
        # adv_target = label
        adv_target = adv_target.to(device)
        adv_target1 = adv_target.unsqueeze(1)
        print('adv', adv_target.shape, adv_target1.shape)
        if args.data_path=='brain':
            mask_logits = adv_target1.repeat(1, n_classes, 1, 1)
            loss_houdini = mask_houdini_loss(mask_logits, num_class=n_classes, device=device)
            # print('dtype', image.dtype)
            # x_adv = semantic_adv.semantic_attk(generator, model,adversary,criterian, image, org_label,
            #                                    adv_target1, adv_target1, targeted)
            x_adv = semantic_adv.semantic_attk(generator, model, adversary, loss_houdini, image, org_label,
                                               adv_target, adv_target1, targeted)
        else:
            with torch.no_grad():
                x_adv = semantic_adv.semantic_attk1(generator, model, adversary, None, image, org_label,
                                                   adv_target, adv_target1, targeted)
        # x_adv = denorm(x_adv)
        out_img = x_adv#* 255
        # print('out', len(out_img), out_img[0].shape)
        with torch.no_grad():
            for k, im in enumerate(out_img):
                # im = im.data.cpu().numpy()
                # im = np.moveaxis(im, 0, -1)
                ind = batch_idx * batch_size + k
                # print('img', ind, im.shape)
                # cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                im = im.flip(0)
                save_image(denorm(im.data.cpu()), os.path.join(adv_dir, '{}.png'.format(ind)))
                adv_imgs.append(im)

    # org_imgs = torch.cat(org_imgs, dim=0)
    # adv_imgs = torch.stack(adv_imgs, dim=0)
    # if 'DAG' in args.attacks:
    #     org_imgs = torch.movedim(org_imgs,1,-1)
    #     adv_imgs = torch.movedim(adv_imgs, 1, -1)
    # print('adv', adv_imgs.size())
    # # if args.channels == 1:
    # #     org_imgs = org_imgs.squeeze(-1)
    # #     adv_imgs = adv_imgs.squeeze(-1)
    # print('shape', org_imgs.size(), adv_imgs.size(), org_imgs.dtype, adv_imgs.dtype)
    # # ssim_val = ssim(org_imgs, adv_imgs, data_range=1, channel=args.channels, size_average=False)
    # ssim_module = SSIM(data_range=1, size_average=False, channel=args.channels)
    # ms_ssim_module = MS_SSIM(data_range=1, size_average=False, channel=args.channels)
    # ssim_val = ssim_module(org_imgs, adv_imgs)
    # avg_ssim = torch.mean(ssim_val).numpy()
    # # ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=1, channel=args.channels, size_average=False)
    # ms_ssim_val = ms_ssim_module(org_imgs, adv_imgs)
    # avg_msssim = torch.mean(ms_ssim_val).numpy()
    # print('avg ssim: ', avg_ssim, 'avg msssim: ', avg_msssim)
    # print('total {} {} images are generated'.format(len(adversarial_examples), args.attacks))
    print('done generating')
    return adversarial_examples

class houdini_loss(nn.Module):
    def __init__(self, use_cuda=True, num_class=3, device='gpu',ignore_index=None):
        super(houdini_loss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.device = device

    def forward(self, logits, target):
        pred = logits.max(1)[1]
        target = target
        # print('pred', pred.shape, target.shape)
        pred_onehot = make_one_hot(pred, self.num_class, self.device)
        # print('tarr', torch.unique(target))
        target_onehot = make_one_hot(target, self.num_class, self.device)
        # print('tar', target_onehot.shape)
        # pred_onehot = Variable(pred_onehot)
        # target_onehot = Variable(target_onehot)
        pred_onehot = pred_onehot.data
        target_onehot = target_onehot.data
        neg_log_softmax = -F.log_softmax(logits, dim=1)
        # print('neg', neg_log_softmax.shape, target_onehot.shape)
        twod_cross_entropy = torch.sum(neg_log_softmax * target_onehot, dim=1)

        pred_score = torch.sum(logits * pred_onehot, dim=1)
        target_score = torch.sum(logits * target_onehot, dim=1)
        # print('sub', sub.min(), sub.max())
        mask = 0.5 + 0.5 * (((pred_score - target_score) / math.sqrt(2)).erf())
        return torch.mean(mask * twod_cross_entropy)

class mask_houdini_loss(nn.Module):
    def __init__(self, mask_logits, num_class=3, weight=10, device='gpu'):
        super(mask_houdini_loss, self).__init__()
        self.houdini = houdini_loss(num_class=num_class, device=device)
        self.mask_logits = mask_logits
        # self.mask_target = mask_target
        self.weight = weight
        # print('mask', mask_logits.shape)

    def forward(self, logits, target):
        # print('ta', torch.unique(target))
        return self.houdini(logits * self.mask_logits,
                            target) * self.weight

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    test_dataset, test_loader = load_data(args)

    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

    print("arg", args.model)
    if args.model == 'UNet':
        model = UNet(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'SegNet':
        model = SegNet(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'DenseNet':
        model = DenseNet(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'AgNet':
        model = AG_Net(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        # model = get_model('AG_Net')
        # model = model(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        model = model.double()
    elif 'deeplab' in args.model:
        model = deeplab.modeling.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride,
                                                      in_channels=args.channels, pretrained_backbone=False)
        if args.separable_conv and 'plus' in args.model:
            deeplab.convert_to_separable_conv(model.classifier)
    model_path = os.path.join(args.model_path,args.data_path,args.model+'.pth')
    print('Load model', model_path)
    device = torch.device(args.device)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    generator = Generator(conv_dim=128, c_dim=1)
    generator_path = os.path.join(args.generator_path)
    generator = generator.to(device)
    print('Load generator', generator_path)
    generator.load_state_dict(torch.load(generator_path))

    Attack(generator, model, test_loader, device, args)

