import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import random
import sys
import os
import cv2

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from model import UNet, SegNet, DenseNet

from dataset.dataset import SampleDataset, AgDataset
from pytorch_msssim import ms_ssim, ssim,  SSIM, MS_SSIM
from dag import DAG
from dag_iqa import cw
# from dag_iqa import DAG
from dag_utils import generate_target, generate_target_swap, generate_target_bs, generate_target_swap_cls
from util import make_one_hot
from attacks.fgsm import i_fgsm, pgd
from attacks.spt import attack
from attacks.square_attack import square_attack_linf, square_attack_l2
from model.AgNet.core.models import AG_Net
from model import deeplab
from model.AgNet.core.utils import get_model
from opts import get_args


def load_data(args):
    
    data_path = args.data_path
    n_classes = args.classes

    # generate loader
    # test_dataset = AgDataset(data_path, n_classes, args.channels, 'adv',args.model,args.attacks, None,'org',
    #                              args.width, args.height)
    test_dataset = AgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                             args.data_type, args.target, 'org', args.width, args.height, args.mask_type,
                             suffix=args.suffix)
    # test_dataset = SampleDataset(data_path, n_classes, args.channels, 'adv', args.mode, args.model, None,\
    #                              'org', args.width, args.height, args.mask_type)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    
    return test_dataset, test_loader

def Attack(model, test_loader, args):
    
    # Hyperparamter for DAG 
    
    num_iterations=20
    # num_iterations = 500
    gamma=0.5
    # gamma = 1

    batch_size = args.batch_size
    device = torch.device(args.device)
        
    model = model.to(device)
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
    print('adv dir', adv_dir)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    targeted = True
    loss = 'margin_loss' if not targeted else 'cross_entropy'
    n_ex, eps, p, n_iter = 1000, 12.75, 0.05, 10000
    hps_str = 'attack={} n_ex={} eps={} p={} n_iter={}'.format(
        'square_linf', n_ex, eps, p, n_iter)
    metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)
    org_imgs, adv_imgs = [],[]
    for batch_idx, (image, label) in enumerate(test_loader):
        # image, label = test_dataset.__getitem__(batch_idx)
        # print('image', image.size())
        # print('label', label.size())
        label = label.squeeze(1)
        # for j, img in enumerate(image):
        #     print('shape',img.shape)
        if args.model == 'AgNet':
            image = image.double()
        else:
            image = image.float()
        org_imgs.append(image)
        # image = image.unsqueeze(0)
        pure_label = label.squeeze(0).numpy()
        pure_image = image.numpy().swapaxes(1,-1)
        # print('un', np.unique(pure_label))
        image , label = image.clone().detach().requires_grad_(True), label.clone().detach().float()
        # image, label = image.clone().detach().requires_grad_(True).long(), label.clone().detach().long()
        image , label = image.to(device), label.to(device)
        # label = label*255
        # label = label.long()
        # unique = torch.unique(label)
        # print('unique', unique.cpu().numpy())
        # for i, cl in enumerate(unique):
        #     label[label == cl] = i
        # Change labels from [batch_size, height, width] to [batch_size, num_classes, height, width]
        # label_oh=make_one_hot(label.long(),n_classes,device)

        label_oh = make_one_hot(label, n_classes, device)
        swapped = True
        print(args.attacks)
        if 'DAG' in args.attacks or 'spt' in args.attacks or 'square' in args.attacks:
            target_class = int(target_class)
            if args.mask_type == '1':
                print('dagA')
                adv_target = torch.zeros_like(label)
                adv_target = make_one_hot(adv_target, n_classes, device)
                print("target ", adv_target.shape, label.shape)

            elif args.mask_type == '2':
                print('dagB')
                adv_target,swapped = generate_target_swap(label_oh.cpu().numpy())
                adv_target=torch.from_numpy(adv_target).float().to(device)
                adv_target=make_one_hot(adv_target, n_classes, device)

            elif args.mask_type == '3':
                print('dagC')
                # choice one randome particular class except background class(0)
                # unique_label = torch.unique(label)
                # target_class = int(random.choice(unique_label[1:]).item())
                # target_class = 2
                adv_target = generate_target_bs(batch_idx, label, target_class=target_class)
                # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
                # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
                adv_target=make_one_hot(adv_target, n_classes, device)
            elif args.mask_type == '4':
                print('dagD')
                # choice one randome particular class except background class(0)
                # unique_label = torch.unique(label)
                # target_class = int(random.choice(unique_label[1:]).item())
                # target_class = 2
                adv_target = generate_target_swap_cls(batch_idx, label, target_class=target_class)
                # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
                # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
                adv_target = make_one_hot(adv_target, n_classes, device)
        elif ('ifgsm' in args.attacks) or ('pgd' in args.attacks)  or ('cw' in args.attacks):
            print('ifgsm,pgd')
            # adv_target = torch.zeros_like(label)
            if args.mask_type=='1':
                # adv_target = torch.zeros_like(label_oh)
                adv_target = torch.zeros_like(label)
                # adv_target = make_one_hot(adv_target, n_classes, device)
                # print('eq', torch.all(torch.eq(adv_target, adv_target_oh)))
            else:
                adv_target, swapped = generate_target_swap(label_oh.cpu().numpy())
                adv_target = torch.from_numpy(adv_target).float()
                # adv_target = make_one_hot(adv_target, n_classes, device)
        else :
            print('else')
            print("wrong adversarial attack types : must be DAG_A, DAG_B, or DAG_C")
            raise SystemExit
        adv_target = adv_target.to(device)
        if 'ifgsm' in args.attacks:
            image_iteration = i_fgsm(idx=batch_idx, model=model,
                                     n_class=args.classes,
                                     x=image,
                                     y=label,
                                     targeted=True,
                                     y_target=adv_target,
                                     iteration=num_iterations,
                                     background_class=0,
                                     device=device,
                                     args=args)
            out_img = image_iteration * 255
            out_img = np.moveaxis(out_img, 1, -1)
            for k, im in enumerate(out_img):
                ind = batch_idx * batch_size + k
                print('img', ind, image_iteration.shape)
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                adv_imgs.append(im)
        elif 'pgd' in args.attacks:
            image_iteration = pgd(idx=batch_idx, model=model,
                                     n_class=args.classes,
                                     x=image,
                                     y=label,
                                     y_target=adv_target,
                                     iteration=num_iterations,
                                     background_class=0,
                                     device=device,
                                     verbose=False)
            out_img = image_iteration * 255
            out_img = np.moveaxis(out_img, 1, -1)
            for k, im in enumerate(out_img):
                ind = batch_idx * batch_size + k
                print('img', ind, image_iteration.shape, out_img.dtype)
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                adv_imgs.append(im)
        elif 'cw' in args.attacks:
            # adv_target = make_one_hot(adv_target, n_classes, device)
            image_iteration = cw(args, idx=batch_idx, model=model,
                                                 image=image,
                                                 ground_truth=label,
                                                 oh_label=label_oh,
                                                 adv_target=adv_target,
                                                 num_iterations=num_iterations,
                                                 gamma=gamma,
                                                 no_background=False,
                                                 background_class=0,
                                                 device=device,
                                                 verbose=False)

            if len(image_iteration) > 0:
                out_img = image_iteration * 255
                out_img = np.moveaxis(out_img, 1, -1)
                for k, im in enumerate(out_img):
                    ind = batch_idx * batch_size + k
                    # print('img', ind, image_iteration.shape, out_img.dtype)
                    cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                    adv_imgs.append(im)
        elif 'DAG' in args.attacks:
            if swapped:
                image_iteration=DAG(args,None, model=model,
                          image=image,
                          ground_truth=label,
                          oh_label=label_oh,
                          adv_target=adv_target,
                          num_iterations=num_iterations,
                          gamma=gamma,
                          no_background=False,
                          background_class=0,
                          device=device,
                          verbose=False)

                out_img1 = image_iteration * 255
                print("ou", out_img1.size())
                if args.channels !=1:
                    out_img = torch.movedim(out_img1, 1, -1)
                else:
                    out_img = out_img1.squeeze(1)
                out_np_img = np.uint8(out_img.detach().cpu().numpy())
                # print('out', out_img[0].size())
                for k, im in enumerate(out_np_img):
                    ind = batch_idx * batch_size + k
                    # print('img', ind, im.shape, out_np_img.dtype)
                    cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                    # adv_imgs.append(out_img1[k])
        elif 'spt' in args.attacks:
            pert_img = attack(image, adv_target, model, device)
            out_img = pert_img * 255
            # print('out', len(out_img), out_img[0].shape)
            for k, im in enumerate(out_img):
                im = im.data.cpu().numpy()
                im = np.moveaxis(im, 0, -1)
                ind = batch_idx * batch_size + k
                print('img', ind, pert_img.shape)
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
                adv_imgs.append(im)
        elif 'square' in args.attacks:
            # print('adv ta', adv_target.shape, image.shape)
            # square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
            square_attack = square_attack_linf
            # square_attack = square_attack_l2
            # Note: we count the queries only across correctly classified images
            print('adv', adv_target.shape)
            n_queries, x_adv = square_attack(model, image, adv_target, eps, n_iter,
                                             p, metrics_path, targeted=targeted, loss_type=loss, device=device,
                                             n_classes=args.classes)
            out_img = x_adv * 255
            # print('out', len(out_img), out_img[0].shape)
            for k, im in enumerate(out_img):
                im = im.data.cpu().numpy()
                im = np.moveaxis(im, 0, -1)
                ind = batch_idx * batch_size + k
                print('img', ind, pert_img.shape)
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(ind)), im)
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

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    test_dataset, test_loader = load_data(args)

    model = None
    print("arg", args.model)
    if args.model == 'UNet':
        model = UNet(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'SegNet':
        model = SegNet(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'DenseNet':
        model = DenseNet(in_channels = n_channels, n_classes = n_classes)

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
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit

    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

    model_path = os.path.join(args.model_path,args.data_path,args.model+'.pth')
    print('Load model', model_path)
    model.load_state_dict(torch.load(model_path))

    Attack(model, test_loader, args)