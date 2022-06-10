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
import glob
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from dataset.dataset import SampleDataset, AgDataset
from skimage.measure import compare_ssim as ssim
from pytorch_msssim import ms_ssim, ssim,  SSIM, MS_SSIM
from attacks.scaling_attack import scl_attack
from opts import get_args


def Attack(args):
    
    adversarial_examples = []
    org_img_dir = '{}/{}/{}/imgs/'.format(args.data_path,args.data_type, args.mode)
    print("org data", org_img_dir)
    adv_dir = './output/scale_attk/{}/{}/'.format(args.data_type, args.mode)
    print('adv dir', adv_dir)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    src_img = cv2.imread('./data/source_imgs/waterm.jpg')
    adv_imgs = []
    for image_path in glob.glob('{}/*.jpg'.format(org_img_dir))+glob.glob('{}/*.png'.format(org_img_dir))\
            +glob.glob('{}/*.bmp'.format(org_img_dir)):
        img_name = image_path.split('/')[-1].replace('bmp', 'png')
        print('img name', img_name)
        image = cv2.imread(image_path)
        # scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.NEAREST
        # scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV
        att_img = scl_attack(src_image=src_img, tar_image=image)
        att_img.save(os.path.join(adv_dir, img_name))
        adv_imgs.append(att_img)
        # if len(image_iteration) >= 1:
        #
        #     adversarial_examples.append([image_iteration[-1],
        #                                  pure_label])
        #
        # del image_iteration
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


def Attack1(args):
    adversarial_examples = []
    org_img_dir = '{}/{}/{}/imgs/'.format(args.data_path, args.data_type, args.mode)
    print("org data", org_img_dir)
    adv_dir = './output/scale_attk/{}/{}/'.format(args.data_type, args.mode)
    print('adv dir', adv_dir)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    src_img = cv2.imread('./scaleatt/data/source_imgs/waterm.jpg')
    npz_file = './{}/{}/{}_{}.npz'.format(args.data_path, args.data_type, args.data_type, args.mode)
    data = np.load(npz_file)['a']
    print('load ', npz_file)
    adv_imgs = []
    for i, image in enumerate(data):
        img_name = '{}.png'.format(i)
        print('img name', i)
        # print('img', image.max())
        image *= 255
        image = np.uint8(image)
        if len(image.shape)==2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        # print('img', image.shape)
        # scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.NEAREST
        # scaling_library: SuppScalingLibraries = SuppScalingLibraries.CV
        att_img = scl_attack(src_image=src_img, tar_image=image)
        att_img.save(os.path.join(adv_dir, img_name))
        adv_imgs.append(att_img)
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


if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes

    # adversarial_examples = Attack(args)
    adversarial_examples = Attack1(args)
    # if args.attack_path is None:
    #
    #     adversarial_path = 'data/' + args.model + '_' + args.attacks + '.pickle'
    #
    # else:
    #     adversarial_path = args.attack_path

    # save adversarial examples([adversarial examples, labels])
    # with open(adversarial_path, 'wb') as fp:
    #     pickle.dump(adversarial_examples, fp)

