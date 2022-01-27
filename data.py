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
        data = np.load(npz_file)
        self.images = data['a']
        self.labels = data['b']
        print('len advset', len(self.adv_images))


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

def augment(imgs, config, train):
    if train:
        np.random.seed(int(time.time() * 1000000) % 1000000000)

    if 'flip' in config and config['flip'] and train:
        stride = np.random.randint(2) * 2 - 1
        for i in range(len(imgs)):
            imgs[i] = imgs[i][:, ::stride, :]

    if 'crop_size' in config:
        crop_size = config['crop_size']
        if train:
            h = np.random.randint(imgs[0].shape[0] - crop_size[0] + 1)
            w = np.random.randint(imgs[0].shape[1] - crop_size[1] + 1)
        else:
            h = int(imgs[0].shape[0] - crop_size[0]) / 2
            w = int(imgs[0].shape[1] - crop_size[1]) / 2
        for i in range(len(imgs)):
            imgs[i] = imgs[i][h:h + crop_size[0], w:w + crop_size[1], :]

    return imgs


def normalize(imgs, net_type):
    if net_type == 'inceptionresnetv2':
        for i in range(len(imgs)):
            imgs[i] = imgs[i].astype(np.float32)
            imgs[i] = 2 * (imgs[i] / 255.0) - 1.0

    else:
        mean = np.asarray([0.485, 0.456, 0.406], np.float32).reshape((1, 1, 3))
        std = np.asarray([0.229, 0.224, 0.225], np.float32).reshape((1, 1, 3))
        for i in range(len(imgs)):
            imgs[i] = imgs[i].astype(np.float32)
            imgs[i] /= 255
            imgs[i] -= mean
            imgs[i] /= std

    return imgs

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
        print('load clean', npz_file)
        data = np.load(npz_file)
        self.labels = data['b']
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