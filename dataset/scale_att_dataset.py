from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import numpy as np
import os
import pickle
import torch
import random as rd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import glob
import re


def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

#for testing and defending
class AttackDataset():
    def __init__(self, root_dir, channels, mode, data_path):
        super(AttackDataset, self).__init__()
        self.data_path = root_dir
        self.adv_dir = './output/scale_attk/{}/{}/'.format(data_path, mode)
        self.label_dir = './data/{}/{}/gt/'.format(data_path,mode)
        print('img dir', self.adv_dir)
        print('label dir', self.label_dir)
        npz_file = './data/{}/scl_attk_{}_{}.npz'.format(self.data_path,self.data_path, mode)
        self.images = []
        self.labels = []

        if not os.path.exists(npz_file):
            print('not found existed dump', npz_file)
            self.images, self.labels = self.read_files(self.adv_dir, self.label_dir, channels, resize=False)
            # print('checklabel', np.unique(data[i][1]))
            np.savez_compressed(npz_file,a=self.images,b=self.labels)
        else:
            print('load ', npz_file)
            data = np.load(npz_file)
            self.images=data['a']
            self.labels=data['b']
            # print('checklabel', np.unique(self.labels[0]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        # print('uni', np.unique(labels))
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = torch_transform(image)
        # labels = torch.from_numpy(labels).unsqueeze(0)
        labels = torch.from_numpy(labels)
        return (image, labels)

    @classmethod
    def read_files(cls, img_dir, label_dir, channels, width=256, height=256, resize=True):
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
                label[label < 100] = 0
                label[label > 150] = 255
                label[(label >= 100) & (label <= 150)] = 128
                label = 255-label
                unique = np.unique(label)
                # print('unique', unique)
                for i, cl in enumerate(unique):
                    label[label == cl] = i
                # print('after', np.unique(label))
                images.append(img)
                labels.append(label)
        return images, labels

