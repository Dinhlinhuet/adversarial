from skimage.metrics import structural_similarity as ssim
import cv2
import os
from pytorch_msssim import ms_ssim
import torch
import numpy as np

data_set = 'fundus'
adv_model = 'SegNet'
# adv_model = 'UNet'
# adv_model = 'DenseNet'
# adv_model = 'AgNet'
# attack = 'DAG_A'
# attack = 'DAG_C'
attack = 'ifgsm'
# target = '0'
# target = '1'
target = '2'
img_dir = '/data/linhld/projects/Pytorch_AdversarialAttacks/data/{}/test/imgs/'.format(data_set)
adv_dir ='/data/linhld/projects/Pytorch_AdversarialAttacks/output/adv/{}/{}/{}/{}/'.format(data_set,adv_model,attack,target)
print('adv', adv_dir)
ssims = []
ms_ssims = []
org_imgs=[]
adv_imgs=[]
for img_name in os.listdir(img_dir):
    org_file = '{}/{}'.format(img_dir, img_name)
    # adv_file = '{}/{}.png'.format(adv_dir, img_name)
    adv_file = '{}/{}.png'.format(adv_dir,int(img_name[1:-4])-1)
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
print('size', org_imgs.size())
ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=255, size_average=False)
avg_msssim = torch.mean(ms_ssim_val).numpy()
print('avg msssim: ', avg_msssim)