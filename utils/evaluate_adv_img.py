from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim, ssim,  SSIM, MS_SSIM
import cv2
import os
from pytorch_msssim import ms_ssim
import torch
import numpy as np
# from piq import FSIM
import piq
from os import path
import sys
import re
print(path.dirname(path.dirname(path.abspath(__file__))))

server = 'linh'
# server = 'linhld'
# data_set = 'fundus'
# data_set = 'octafull'
# data_set = 'brain'
data_set = 'lung'
def evaluate_adv():
    # attack = 'cw'
    # adv_model = 'SegNet'
    # adv_model = 'UNet'
    # adv_model = 'DenseNet'
    # adv_model = 'AgNet'
    attack = 'dag'
    # attack = 'ifgsm'
    # target = '0'
    # target = '1'
    # target = '2'
    # proj_dir = '/data/{}/projects/Pytorch_AdversarialAttacks/'.format(server)
    proj_dir = path.dirname(path.dirname(path.abspath(__file__)))
    suffix = 'ce0'
    adv_models = ['AgNet', 'UNet', 'DenseNet','deeplabv3plus_resnet101']
    # adv_models = ['SegNet']
    if attack=='ifgsm':
        cbs = ['m1t0']
    elif 'dag' in attack:
            cbs = [('DAG_A', 'm1t0'), ('DAG_D', 'm4t0')]
    ssims = []
    fsims = []
    org_imgs = []
    adv_imgs = []
    for cb in cbs:
        for adv_model in adv_models:
            img_dir = '{}/data/{}/test/imgs/'.format(proj_dir, data_set)
            # adv_dir ='/data/linhld/projects/Pytorch_AdversarialAttacks/output/adv/{}/{}/{}/{}/'.format(data_set,adv_model,attack,target)
            if attack=='ifgsm':
                adv_dir = '{}/output/adv/{}/test/{}/{}/{}/'.format(proj_dir, data_set, adv_model, attack, cb)
            else:
                adv_dir ='{}/output/adv/{}/test/{}/{}/{}/{}/'.format(proj_dir, data_set,adv_model,attack,cb[0], cb[1])
            print('adv dir', adv_dir)
            # ssims = []
            # org_imgs=[]
            # adv_imgs=[]
            for img_name in os.listdir(img_dir):
                org_file = '{}/{}'.format(img_dir, img_name)
                # print('img',)
                if data_set!='fundus':
                    adv_file = '{}/{}'.format(adv_dir, img_name)
                else:
                    adv_file = '{}/{}.png'.format(adv_dir,filename(img_name)-1)
                org_img = cv2.imread(org_file)
                adv_img = cv2.imread(adv_file)
                if org_img is None:
                    print('org')
                if adv_img is None:
                    print('adv', adv_file)
                ssim_noise = ssim(org_img, adv_img,
                                  data_range=255,multichannel=True)
                ssims.append(ssim_noise)
                org_imgs.append(org_img)
                adv_imgs.append(adv_img)
                org_img=torch.from_numpy(np.array(np.moveaxis(org_img, -1, 0))).type(torch.FloatTensor).unsqueeze(0)
                adv_img = torch.from_numpy(np.array(np.moveaxis(adv_img, -1, 0))).type(torch.FloatTensor).unsqueeze(0)
                fsim_val = piq.fsim(org_img, adv_img, data_range=255., reduction='none')
                fsims.append(fsim_val)
                # print(ssim_noise)
    avg_ssim = round(sum(ssims)/len(ssims),3)
    print('avg ssim: ', avg_ssim)
    # org_imgs = torch.from_numpy(np.array(np.moveaxis(org_imgs,-1,1))).type(torch.FloatTensor)
    # adv_imgs = torch.from_numpy(np.array(np.moveaxis(adv_imgs,-1, 1))).type(torch.FloatTensor)
    # print('size', org_imgs.size())
    # ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=255, size_average=False)
    # avg_msssim = torch.mean(ms_ssim_val).numpy()
    # print('avg msssim: ', avg_msssim)
    avg_fsim = torch.mean(fsim_val).numpy()
    print('fsim', avg_fsim)

def evaluate_attack():
    attack = 'scale_attk'
    # proj_dir = '/data/{}/projects/Pytorch_AdversarialAttacks/'.format(server)
    proj_dir = '/home/{}/projects/Pytorch_AdversarialAttacks/'.format(server)
    mode = 'test'
    img_dir = '{}data/{}/test/imgs/'.format(proj_dir, data_set)
    adv_dir ='{}/output/{}/{}/{}/'.format(proj_dir, attack, data_set,mode)
    print('adv dir', adv_dir)
    print('img dir', img_dir)
    ssims = []
    ms_ssims = []
    org_imgs=[]
    adv_imgs=[]
    for img_name in os.listdir(img_dir):
        org_file = '{}/{}'.format(img_dir, img_name)
        img_name = img_name.replace('bmp','png')
        adv_file = '{}/{}'.format(adv_dir,img_name)
        org_img = cv2.imread(org_file)
        adv_img = cv2.imread(adv_file)
        if org_img is None:
            print('org')
        if adv_img is None:
            print('adv', adv_file)
        ssim_noise = ssim(org_img, adv_img,
                          data_range=255,multichannel=True)
        ssims.append(ssim_noise)
        # org_imgs=np.append(org_imgs,org_img)
        # adv_imgs=np.append(adv_imgs,adv_img)
        org_imgs.append(org_img)
        adv_imgs.append(adv_img)
        # print(ssim_noise)
    avg_ssim = round(sum(ssims)/len(ssims),3)
    print('avg ssim: ', avg_ssim)
    org_imgs = torch.from_numpy(np.array(np.moveaxis(org_imgs,-1,1))).type(torch.FloatTensor)
    adv_imgs = torch.from_numpy(np.array(np.moveaxis(adv_imgs,-1, 1))).type(torch.FloatTensor)
    # print('size', org_imgs.size())
    ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=255, size_average=False)
    avg_msssim = torch.mean(ms_ssim_val).numpy()
    print('avg msssim: ', avg_msssim)
    fsim_val = piq.fsim(org_imgs, adv_imgs, data_range=255., reduction='none')
    avg_fsim = torch.mean(fsim_val).numpy()
    print('fsim', avg_fsim)

def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

if __name__=='__main__':
    evaluate_attack()
    # evaluate_adv()