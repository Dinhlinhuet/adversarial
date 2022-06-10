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
import glob
import re


def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

class SampleDataset(Dataset):
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, type, target_class, data_type, width,
                 height,mask_type, suffix):
        super(Dataset, self).__init__()
        self.num_classes = num_class
        self.data_path = root_dir
        # train_img_dir = './data/{}/train/imgs/'.format(root_dir)
        # val_img_dir = './data/{}/val/imgs/'.format(root_dir)
        # test_img_dir = './data/{}/test/imgs/'.format(root_dir)
        # train_label_dir = './data/{}/train/labels/'.format(root_dir)
        # val_label_dir = './data/{}/val/labels/'.format(root_dir)
        # test_label_dir = './data/{}/test/labels/'.format(root_dir)
        out_img_dir = './data/{}/{}/img/'.format(root_dir, mode)
        out_label_dir = './data/{}/{}/gt/'.format(root_dir, mode)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_label_dir):
            os.makedirs(out_label_dir)
        print('mode', mode)
        root = './data/{}/'.format(self.data_path)
        # root = './data/lung_new/'
        self.img_dir = './data/eye/{}/imgs/'.format(mode)
        self.label_dir = './data/eye/{}/gt/'.format(mode)
        if mode=='train':
            if data_type == 'org':
                npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            else:
                npz_org_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
                npz_file = './data/{}/denoiser/{}/{}_adv_train_{}_{}_m{}t{}.npz'.format(self.data_path, suffix, self.data_path,
                                                                                model,
                                                                                data_type,
                                                                                mask_type, target_class)
                self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type,
                                                                            mask_type,
                                                                            target_class)
        elif mode == 'val':
            npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
        elif mode=='test':
            if data_type=='org':
                npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
                # npz_file = './data/{}/512/{}_test.npz'.format(self.data_path, self.data_path)
            else:
                if self.data_path != 'fundus':
                    # npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path,suffix, self.data_path, model, data_type,
                    #                                                    mask_type, target_class)
                    npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path, suffix, self.data_path, model,
                                                                             type,
                                                                             mask_type, target_class)
                else:
                    npz_file = './data/{}/{}/{}_adv_{}_m{}t{}.npz'.format(self.data_path, suffix, self.data_path,
                                                                              data_type,mask_type, target_class)
                # npz_file = './data/{}/{}/{}_adv_{}_{}_{}_m{}t{}.npz'.format(self.data_path, suffix, self.data_path, model,
                #                                                          type, data_type,
                #                                                          mask_type, target_class)
                # npz_file = './data/{}/{}_adv_{}_mix_label.npz'.format(self.data_path, self.data_path, type)
                # npz_file = './data/{}/{}_adv_{}_pure_target.npz'.format(self.data_path, self.data_path, type)
                # elif target_class:
                #     npz_file = './data/{}/{}_adv_{}_{}_{}.npz'.format(self.data_path,self.data_path, model, type, target_class)
                # else:
                #     npz_file = './data/{}/{}_adv_{}_{}.npz'.format(self.data_path,suffix, self.data_path, model, type
                #                                                       )
                if not os.path.exists('./data/{}/{}/'.format(self.data_path,suffix)):
                    os.makedirs('./data/{}/{}/'.format(self.data_path,suffix))
                npz_org_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # self.img_dir = './output/adv/{}/{}/{}/'.format(self.data_path,model, type)
                # if mask_type != '':
                # self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, type, mask_type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type, mask_type,
                #                                                          target_class)
                if self.data_path != 'fundus':
                    self.img_dir = './output/adv/{}/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, type,data_type,
                                                                         mask_type,
                                                                         target_class)

                # else:
                #     self.img_dir = './output/adv/{}/{}/{}/{}/'.format(self.data_path, model, type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/'.format('fundus_512', model, type, target_class)
        else:
            npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
            self.img_dir = './data/{}/test/imgs/'.format(self.data_path)
            # npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/train/imgs/'.format(self.data_path)
            # npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/val/imgs/'.format(self.data_path)
        self.images = []
        self.labels = []
        # cm = plt.cm.jet
        # cm = plt.get_cmap('gist_rainbow', 1000)
        # cm= plt.get_cmap('viridis', 28)
        # print(self.img_dir)
        if data_type=='org':
            #train and test org, generate adv
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
                # print('writing')
                # for i, img in enumerate(self.images):
                #     cv2.imwrite('./{}/{}.png'.format(out_img_dir, i), img*255)
                # for i, img in enumerate(self.labels):
                #     cv2.imwrite('./{}/{}.png'.format(out_label_dir, i), img*255)
        else:
            #test adv mode
            if not os.path.exists(npz_file):
                test_org_data = np.load(npz_org_file)
                self.labels = test_org_data['b']
                print('not found existed dump adv',npz_file)
                ls_names = sorted(os.listdir(self.img_dir),key=filename)
                print('ls name', len(ls_names))
                for img_name in ls_names:
                    if 'png' or 'bmp' in img_name:
                        img_path = os.path.join(self.img_dir, img_name)
                        # print('label', label_path)
                        if channels == 1:
                            img = cv2.imread(img_path,0)
                        else:
                            img = cv2.imread(img_path)
                        img = cv2.resize(img,(256,256))/ 255
                        # img = img/255
                        self.images.append(img)
                        # print('checklabel', np.unique(data[i][1]))
                np.savez_compressed(npz_file, a=self.images, b=self.labels)
            else:
                print('load ',npz_file)
                data = np.load(npz_file)
                print(data.files)
                self.images = data['a']
                print(self.images.dtype)
                # self.labels = data['b']
                test_org_data = np.load(npz_org_file)
                self.images = test_org_data['a']
                self.labels = test_org_data['b']
                # # print('writing')
                # for i, img in enumerate(self.images):
                #     cv2.imwrite('./{}/{}.png'.format(out_img_dir, i), img*255)
                # for i, img in enumerate(self.labels):
                #     cv2.imwrite('./{}/{}.png'.format(out_label_dir, i), img*255)
        print('len imgs', len(self.images))
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

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