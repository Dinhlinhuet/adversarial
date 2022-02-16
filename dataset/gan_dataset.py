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

class GanDataset(Dataset):
    def __init__(self, root_dir, channels, mode, width,
                 height):
        super(Dataset, self).__init__()
        self.data_path = root_dir
        out_img_dir = './data/{}/{}/imgs/'.format(root_dir, mode)
        out_label_dir = './data/{}/{}/gt/'.format(root_dir, mode)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_label_dir):
            os.makedirs(out_label_dir)
        root = './data/{}/'.format(self.data_path)
        self.img_dir = './data/eye/{}/imgs/'.format(mode)
        self.label_dir = './data/eye/{}/gt/'.format(mode)
        if mode=='train':
            npz_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
        elif mode == 'val':
            npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
        elif mode=='test':
            npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
        else:
            npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
            self.img_dir = './data/{}/test/imgs/'.format(self.data_path)
        self.images = []
        self.labels = []

        if not os.path.exists(npz_file):
            print('not found existed dump', npz_file)
            self.images, self.labels = self.read_files(self.img_dir, self.label_dir, out_img_dir, out_label_dir,
                                                       channels, width, height)
            # print('checklabel', np.unique(data[i][1]))
            np.savez_compressed('{}/{}_{}.npz'.format(root,self.data_path, mode),a=self.images,b=self.labels)
        else:
            print('load ', npz_file)
            data = np.load(npz_file)
            self.images=data['a']
            self.labels=data['b']
            # print('checklabel', np.unique(self.labels[0]))
        self.images = self.images.astype(np.float32)
        self.labels = np.expand_dims(self.labels,1)
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def read_files(self, img_dir, label_dir, out_img_dir, out_label_dir, channels, width, height):
        images=[]
        labels=[]
        ls_names = os.listdir(label_dir)
        ls_names = sorted(ls_names, key=filename)
        for img_name in ls_names:
            if 'png' or 'bmp' or 'jpg' in img_name:
                # print(self.img_dir,img_name[:-9]+'.png')
                img_path = os.path.join(img_dir, img_name.replace('_mask', ''))
                label_path = os.path.join(label_dir, img_name)
                # print('label', label_path)
                # print('img path', img_path)
                if channels == 1:
                    org_img = cv2.imread(img_path, 0)
                # print("imgname", img_name)
                else:
                    org_img = cv2.imread(img_path)
                org_img = cv2.resize(org_img, (height, width))
                img = org_img / 255
                label = cv2.imread(label_path, 0)
                # label = 255-label
                label[label < 100] = 0
                label[label > 150] = 255
                label[(label >= 100) & (label <= 150)] = 128
                label = cv2.resize(label, (height, width), interpolation=cv2.INTER_NEAREST)
                # cv2.imwrite('{}/{}'.format(out_img_dir, img_name), org_img)
                # cv2.imwrite('{}/{}'.format(out_label_dir, img_name), label)
                unique = np.unique(label)
                # print('unique', unique)
                for i, cl in enumerate(unique):
                    label[label == cl] = i
                # print('after', np.unique(label))
                images.append(img)
                labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        # print('uni', np.unique(labels))

        image = self.torch_transform(image)
        # labels = torch.from_numpy(labels).unsqueeze(0)
        labels = torch.from_numpy(labels)
        # labels = torch_transform(labels)
        # print('ll', labels.size())
        # print('torh', torch.unique(labels).cpu().numpy())
        return (image, labels)