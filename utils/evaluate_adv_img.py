from skimage.metrics import structural_similarity as ssim
import cv2
import os
from pytorch_msssim import ms_ssim
import torch
import numpy as np
# from piq import FSIM
import piq

# server = 'linh'
server = 'linhld'
# data_set = 'fundus'
data_set = 'octafull'
# data_set = 'brain'
def evaluate_adv():
    attack = 'cw'
    # adv_model = 'SegNet'
    # adv_model = 'UNet'
    # adv_model = 'DenseNet'
    # adv_model = 'AgNet'
    # attack = 'DAG_A'
    # attack = 'DAG_C'
    # attack = 'ifgsm'
    # target = '0'
    # target = '1'
    # target = '2'
    proj_dir = '/data/{}/projects/Pytorch_AdversarialAttacks/'.format(server)
    suffix = 'ce0'
    adv_models = ['SegNet', 'UNet', 'DenseNet']
    cbs = [('1',1),('3',1),('3',2)]
    for cb in cbs:
        mask, target = cb
        for adv_model in adv_models:
            img_dir = '{}data/{}/test/imgs/'.format(proj_dir, data_set)
            # adv_dir ='/data/linhld/projects/Pytorch_AdversarialAttacks/output/adv/{}/{}/{}/{}/'.format(data_set,adv_model,attack,target)
            adv_dir ='{}/output/adv/{}/test/{}/{}/m{}t{}/'.format(proj_dir, data_set,adv_model,attack,mask, target)
            print('adv dir', adv_dir)
            ssims = []
            ms_ssims = []
            org_imgs=[]
            adv_imgs=[]
            for img_name in os.listdir(img_dir):
                org_file = '{}/{}'.format(img_dir, img_name)
                if data_set=='octa3mfull':
                    adv_file = '{}/{}'.format(adv_dir, img_name)
                else:
                    adv_file = '{}/{}.png'.format(adv_dir,int(img_name[1:-4])-1)
                org_img = cv2.imread(org_file)
                adv_img = cv2.imread(adv_file)
                if org_img is None:
                    print('org')
                if adv_img is None:
                    print('adv', adv_file)
                # ssim_noise = ssim(org_img, adv_img,
                #                   data_range=255,multichannel=True)
                # ssims.append(ssim_noise)
                # org_imgs=np.append(org_imgs,org_img)
                # adv_imgs=np.append(adv_imgs,adv_img)
                org_imgs.append(org_img)
                adv_imgs.append(adv_img)
                # print(ssim_noise)
            # avg_ssim = round(sum(ssims)/len(ssims),3)
            # print('avg ssim: ', avg_ssim)
            org_imgs = torch.from_numpy(np.array(np.moveaxis(org_imgs,-1,1))).type(torch.FloatTensor)
            adv_imgs = torch.from_numpy(np.array(np.moveaxis(adv_imgs,-1, 1))).type(torch.FloatTensor)
            # print('size', org_imgs.size())
            # ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=255, size_average=False)
            # avg_msssim = torch.mean(ms_ssim_val).numpy()
            # print('avg msssim: ', avg_msssim)
            fsim_val = piq.fsim(org_imgs, adv_imgs, data_range=255., reduction='none')
            avg_fsim = torch.mean(fsim_val).numpy()
            print('fsim', avg_fsim)

def evaluate_attack():
    attack = 'scale_attck'
    proj_dir = '/data/{}/projects/Pytorch_AdversarialAttacks/'.format(server)
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

if __name__=='__main__':
    evaluate_attack()