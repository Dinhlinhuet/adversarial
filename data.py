import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
#from scipy.misc import cv2.imread, imresize
import cv2
import time
import random as rd
import re

def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

class DefenseDataset(Dataset):
    def __init__(self, data_path, phase, channels):
        assert phase == 'train' or phase == 'val' or phase == 'test'
        self.phase = phase
        # self.dataset = dataset
        # self.attack = attack
        self.data_path = data_path
        self.images=[]
        self.adv_npz_list = []
        npz_file = './data/{}/{}_{}.npz'.format(data_path,data_path, phase)
        # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}.npz'.format(data_path, data_path, phase, 'SegNet', 'DAG_C')
        # octa3mfull
        # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path,data_path,phase, 'UNet', 'DAG_A', '0')
        #fundus
        adv_list = []
        combine = []
        if data_path == 'fundus':
            for cls in ['m1','m2']:
                combine.append(('ifgsm',cls))
                # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path, data_path, phase, 'UNet', 'ifgsm',
                #                                                               cls)
                # adv_list.append(adv_npz_file)
        elif data_path == 'octa3mfull':
            for cls in ['1','2']:
                combine.append(('DAG_C', cls))
                # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path,data_path,phase, 'UNet', 'DAG_C', cls)
                # adv_list.append(adv_npz_file)
            combine.append(('DAG_A', '0'))
            # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path, data_path, phase, 'UNet',
            #                                                                   'DAG_A', '0')
            # adv_list.append(adv_npz_file)

        #brain
        # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path, data_path, phase, 'DenseNet', 'ifgsm',
        #                                                                   '1')
        # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path, data_path, phase, 'UNet',
        #                                                                   'pgd','1')
        if not os.path.exists('./data/{}/denoiser/'.format(data_path)):
            os.mkdir('./data/{}/denoiser/'.format(data_path))
        for comb in combine:
            adv_images = []
            adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(data_path, data_path, phase, 'UNet',
                                                                              comb[0],comb[1])
            adv_list.append(adv_npz_file)
            if not os.path.exists(adv_npz_file):
                print("not found", adv_npz_file)
                #fundus
                adv_dir = './output/adv/{}/{}/{}/{}/{}/'.format(data_path, phase, 'UNet', comb[0],comb[1])
                # adv_dir = './output/adv/{}/{}/{}/{}/{}/'.format(data_path, phase, 'UNet', 'pgd','1')
                # brain
                # adv_dir = './output/adv/{}/{}/{}/{}/{}/'.format(data_path, phase, 'UNet', 'pgd','1')
                # adv_dir = './output/adv/{}/{}/{}/{}/{}/'.format(data_path, phase, 'UNet', 'DAG_A', '0')
                # adv_dir = './output/adv/{}/{}/{}/{}/'.format(data_path, phase, 'SegNet', 'DAG_C')
                def filename(x):
                    return int(x[:-4])

                ls_names = sorted(os.listdir(adv_dir), key=filename)
                # print('lsname', ls_names)
                for img_name in ls_names:
                    if 'png' in img_name:
                        img_path = os.path.join(adv_dir, img_name)
                        # print('label', label_path)
                        if channels == 1:
                            img = cv2.imread(img_path, 0)
                        else:
                            img = cv2.imread(img_path)
                        # img = cv2.resize(img,(256,256))/ 255
                        img = img / 255
                        adv_images.append(img)
                        # print('checklabel', np.unique(data[i][1]))
                np.savez_compressed(adv_npz_file, a=adv_images)
            else:
                print('load adv', adv_npz_file)
                adv_data = np.load(adv_npz_file)
                adv_images = adv_data['a']
            self.adv_npz_list.append(adv_images)
        print('load clean', npz_file)
        data = np.load(npz_file)
        self.images = data['a']
        self.labels = data['b']


    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]

        num_adv_type = len(self.adv_npz_list)-1
        i = rd.randint(0,num_adv_type)
        adv_file = self.adv_npz_list[i]
        adv_image = adv_file[index]
        # adv_labels = self.adv_labels[index]

        # cv2.imwrite('./data/brain/{}.png'.format(index), image * 255)
        # cv2.imwrite('./data/{}/test/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/test/labels/{}.png'.format(self.data_path,index), labels * 255)
        # cv2.imwrite('./data/{}/train/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/train/labels/{}.png'.format(self.data_path,index), labels * 255)
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = torch_transform(image)
        # labels = torch.from_numpy(labels).unsqueeze(0)
        # print('label', labels.size())
        # labels = torch_transform(labels)
        adv_image = torch_transform(adv_image)

        return image, adv_image, labels

    def __len__(self):
        return len(self.images)

class DefenseSclDataset(Dataset):
    def __init__(self, data_path, phase, channels):
        assert phase == 'train' or phase == 'val' or phase == 'test'
        self.phase = phase
        # self.dataset = dataset
        # self.attack = attack
        self.data_path = data_path
        npz_file = './data/{}/{}_{}.npz'.format(data_path,data_path, phase)

        adv_data_dir = './data/{}/denoiser/'.format(data_path)
        if not os.path.exists(adv_data_dir):
            os.makedirs(adv_data_dir)
        adv_npz_file = '{}/scl_attk_{}_{}.npz'.format(adv_data_dir, data_path, phase)
        adv_dir = './output/scale_attk/{}/{}/'.format(data_path, phase)
        self.adv_images = []
        if not os.path.exists(adv_npz_file):
            print('not found', adv_npz_file)
            def filename(x):
                return int(re.sub('[^0-9]','', x.split('.')[0]))

            ls_names = sorted(os.listdir(adv_dir), key=filename)
            # print('lsname', ls_names)
            for img_name in ls_names:
                # print('img', img_name)
                img_path = os.path.join(adv_dir, img_name)
                # print('label', label_path)
                if channels == 1:
                    img = cv2.imread(img_path, 0)
                else:
                    img = cv2.imread(img_path)
                # img = cv2.resize(img,(256,256))/ 255
                img = img / 255
                self.adv_images.append(img)
            np.savez_compressed(adv_npz_file, a=self.adv_images)
        else:
            print('load adv', adv_npz_file)
            adv_data = np.load(adv_npz_file)
            self.adv_images = adv_data['a']
        print('load clean', npz_file)
        if 'octa' not in data_path:
            data = np.load(npz_file)
            self.images = data['a']
            self.labels = data['b']
        else:
            self.img_dir = './data/{}/{}/imgs/'.format(data_path, phase)
            self.label_dir = './data/{}/{}/gt/'.format(data_path, phase)
            self.images, self.labels = read_files(self.img_dir, self.label_dir, channels, resize=False)
            np.savez_compressed(npz_file, a=self.images, b=self.labels)

        print('len advset', len(self.adv_images))
        self.debug_cln_dir = './output/debug/clean/'
        self.debug_attk_dir = './output/debug/attk/'
        for i, img in enumerate(self.images):
            cv2.imwrite(os.path.join(self.debug_cln_dir, '{}.png'.format(i)), img*255)
            cv2.imwrite(os.path.join(self.debug_attk_dir, '{}.png'.format(i)), self.adv_images[i]*255)

    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        adv_image = self.adv_images[index]
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = torch_transform(image)
        # labels = torch.from_numpy(labels).unsqueeze(0)
        # print('label', labels.size())
        # labels = torch_transform(labels)
        adv_image = torch_transform(adv_image)

        return image, adv_image, labels

    def __len__(self):
        return len(self.images)

class DefenseSclTestDataset(Dataset):
    def __init__(self, data_path, phase, channels):
        assert phase == 'train' or phase == 'val' or phase == 'test'
        self.phase = phase
        # self.dataset = dataset
        # self.attack = attack
        self.data_path = data_path
        npz_file = './data/{}/{}_{}.npz'.format(data_path,data_path, phase)

        # adv_npz_file = './data/{}/denoiser/scl_attk_{}_{}.npz'.format(data_path, data_path, phase)
        adv_npz_file = './data/{}/scl_attk_{}_{}.npz'.format(data_path, data_path, phase)
        adv_dir = './output/scale_attk/{}/{}/'.format(data_path, phase)
        self.adv_images = []
        if not os.path.exists(adv_npz_file):
            print('not found', adv_npz_file)
            def filename(x):
                return int(re.sub('[^0-9]','', x.split('.')[0]))

            ls_names = sorted(os.listdir(adv_dir), key=filename)
            # print('lsname', ls_names)
            for img_name in ls_names:
                # print('img', img_name)
                img_path = os.path.join(adv_dir, img_name)
                # print('label', label_path)
                if channels == 1:
                    img = cv2.imread(img_path, 0)
                else:
                    img = cv2.imread(img_path)
                # img = cv2.resize(img,(256,256))/ 255
                img = img / 255
                self.adv_images.append(img)
            np.savez_compressed(adv_npz_file, a=self.adv_images)
        else:
            print('load adv', adv_npz_file)
            adv_data = np.load(adv_npz_file)
            self.adv_images = adv_data['a']
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            print('load clean', npz_file)
            self.labels = data['b']
        else:
            self.img_dir = './data/{}/{}/imgs/'.format(data_path, phase)
            self.label_dir = './data/{}/{}/gt/'.format(data_path, phase)
            self.images, self.labels = read_files(self.img_dir, self.label_dir, channels, resize=False)
            np.savez_compressed(npz_file, a=self.images, b=self.labels)
        print('len advset', len(self.adv_images))


    def __getitem__(self, index):
        # print("idx", index)
        labels = self.labels[index]
        adv_image = self.adv_images[index]
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        adv_image = torch_transform(adv_image)

        return adv_image, labels

    def __len__(self):
        return len(self.labels)

def read_files(img_dir, label_dir, channels, width=256, height=256, resize=True):
    images=[]
    labels=[]
    ls_names = os.listdir(label_dir)
    ls_names = sorted(ls_names, key=filename)
    for img_name in ls_names:
        if 'png' or 'bmp' or 'jpg' in img_name:
            # print(self.img_dir,img_name[:-9]+'.png')
            img_path = os.path.join(img_dir, img_name.replace('_mask', ''))#.replace('bmp','png'))
            label_path = os.path.join(label_dir, img_name)
            # print('label', label_path)
            # print('img path', img_path)
            if channels == 1:
                org_img = cv2.imread(img_path, 0)
            # print("imgname", img_name)
            else:
                org_img = cv2.imread(img_path)
            label = cv2.imread(label_path, 0)
            if resize:
                org_img = cv2.resize(org_img, (height, width))
                label = cv2.resize(label, (height, width), interpolation=cv2.INTER_NEAREST)
            img = org_img / 255
            images.append(img)
            labels.append(label)
    return images, labels