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

class SampleDataset(Dataset):
    def __init__(self, root_dir, num_class, channels, mode, model=None, type=None, target_class=None, data_type='org',
        width = 256, height = 256):
        self.num_classes = num_class
        self.data_path = root_dir
        train_img_dir = './data/{}/train/imgs/'.format(root_dir)
        test_img_dir = './data/{}/test/imgs/'.format(root_dir)
        train_label_dir = './data/{}/train/gt/'.format(root_dir)
        test_label_dir = './data/{}/test/gt/'.format(root_dir)
        # test_folder = './data/{}/512/test/'.format(root_dir)
        if not os.path.exists(train_img_dir):
            os.makedirs(train_img_dir)
        if not os.path.exists(test_img_dir):
            os.makedirs(test_img_dir)
        if not os.path.exists(train_label_dir):
            os.makedirs(train_label_dir)
        if not os.path.exists(test_label_dir):
            os.makedirs(test_label_dir)
        # root = './data/OCTA-500/OCTA_3M/'
        root = './data/{}/'.format(self.data_path)
        self.img_dir = './data/OCTA-500/{}/imgs/'.format(self.data_path)
        self.label_dir = './data/OCTA-500/{}/gt/'.format(self.data_path)
        if mode=='train':
            npz_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
        elif mode=='test':
            if data_type=='org':
                npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
                # npz_file = './data/{}/512/{}_test.npz'.format(self.data_path, self.data_path)
            else:
                npz_file = './data/{}/{}_adv_{}_{}_{}.npz'.format(self.data_path,self.data_path, model, type, target_class)
                npz_org_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # self.img_dir = './output/adv/{}/{}/{}/'.format(self.data_path,model, type)
                self.img_dir = './output/adv/{}/{}/{}/{}/'.format(self.data_path, model, type, target_class)
        else:
            npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
            self.img_dir = './data/{}/test/imgs/'.format(self.data_path,self.data_path)
            # npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/train/imgs/'.format(self.data_path, self.data_path)
        self.images = []
        self.labels = []

        ratio = 0.85
        # data form [images, labels]
        # with open(self.data_path, 'rb') as fp:
        #     data = pickle.load(fp)

        # cm = plt.cm.jet
        # cm = plt.get_cmap('gist_rainbow', 1000)
        # cm= plt.get_cmap('viridis', 28)
        np.random.seed(1)
        print(self.img_dir)
        if data_type=='org':
            #train and test org, generate adv
            if not os.path.exists(npz_file):
                print('not found existed dump', npz_file)
                ls_name = os.listdir(self.label_dir)
                total_img = len(ls_name)
                idc = np.zeros((total_img))
                idc[:int(total_img*ratio)]=1
                np.random.shuffle(idc)
                images = []
                labels = []
                i=0
                for idx, train in enumerate(idc):
                    img_name = ls_name[idx]
                    if 'png' or 'bmp' in img_name:
                        # print(self.img_dir,img_name[:-9]+'.png')
                        img_path = os.path.join(self.img_dir,img_name.replace('_mask',''))
                        label_path = os.path.join(self.label_dir,img_name)
                        # print('label', label_path)
                        # print('img path', img_path)
                        if channels == 1:
                            org_img = cv2.imread(img_path, 0)
                        # print("imgname", img_name)
                        else:
                            org_img = cv2.imread(img_path)
                        org_img = cv2.resize(org_img,(width,height))
                        img= org_img/255
                        label = cv2.imread(label_path,0)
                        label = cv2.resize(label, (width, height),  interpolation= cv2.INTER_NEAREST)
                        label[label < 100] = 0
                        label[label > 150] = 255
                        label[(label >= 100) & (label <= 150)] = 128
                        unique = np.unique(label)
                        for i, cl in enumerate(unique):
                            label[label == cl] = i
                        # print('after', np.unique(label))
                        if train:
                            self.images.append(img)
                            self.labels.append(label)
                            cv2.imwrite('{}/{}'.format(train_img_dir, img_name), org_img)
                            cv2.imwrite('{}/{}'.format(train_label_dir, img_name), label)
                        else:
                            images.append(img)
                            labels.append(label)
                            # cv2.imwrite('{}/{}.png'.format(test_folder,i), org_img)
                            i+=1
                            cv2.imwrite('{}/{}'.format(test_img_dir, img_name), org_img)
                            cv2.imwrite('{}/{}'.format(test_label_dir, img_name), label)
                        # print('checklabel', np.unique(data[i][1]))
                np.savez_compressed('{}/{}_train.npz'.format(root,self.data_path),a=self.images,b=self.labels)
                np.savez_compressed('{}/{}_test.npz'.format(root,self.data_path),a=images,b=labels)
            else:
                #generate adv
                print('load ', npz_file)
                data = np.load(npz_file)
                self.images=data['a']
                self.labels=data['b']
        else:
            #test adv mode
            if not os.path.exists(npz_file):
                test_org_data = np.load(npz_org_file)
                self.labels = test_org_data['b']
                print('not found existed dump adv',npz_file)
                ls_names = sorted(os.listdir(self.img_dir),key=filename)
                for img_name in ls_names:
                    if 'png' or 'bmp' in img_name:
                        img_path = os.path.join(self.img_dir, img_name)
                        # print('label', label_path)
                        img = cv2.imread(img_path,0)
                        # img = cv2.resize(img,(256,256))/ 255
                        img = img/255
                        self.images.append(img)
                        # print('checklabel', np.unique(data[i][1]))
                np.savez_compressed(npz_file, a=self.images, b=self.labels)
            else:
                print('load ',npz_file)
                data = np.load(npz_file)
                self.images = data['a']
                self.labels = data['b']

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        # print('uni', np.unique(labels))
        # cv2.imwrite('./data/brain/{}.png'.format(index), image * 255)
        # cv2.imwrite('./data/{}/test/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/test/labels/{}.png'.format(self.data_path,index), labels * 255)
        # cv2.imwrite('./data/{}/test/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/test/labels/{}.png'.format(self.data_path,index), labels * 255)
        # cv2.imwrite('./data/{}/train/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/train/labels/{}.png'.format(self.data_path,index), labels * 255)
        # print('uniqe', np.unique(labels*255))
        # cv2.imwrite('./data/{}/train/imgs/{}.png'.format(self.data_path,index), image * 255)
        # cv2.imwrite('./data/{}/train/labels/{}.png'.format(self.data_path,index), labels * 255)
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = torch_transform(image)
        labels = torch.from_numpy(labels).unsqueeze(0)
        # labels = torch.from_numpy(labels)
        # labels = torch_transform(labels)
        # print('ll', labels.size())
        # print('torh', torch.unique(labels).cpu().numpy())
        return (image, labels)

class SegmentDataset(SampleDataset):
# # use only for train combine
#     def __init__(self, root_dir, num_class, channels, mode, model, type, target_class, data_type, width, height):
#         self.num_classes = num_class
#         self.data_path = root_dir
#         # test_folder = './data/{}/test/'.format(root_dir)
#         # test_folder = './data/{}/512/test/'.format(root_dir)
#         # if not os.path.exists(test_folder):
#         #     os.makedirs(test_folder)
#         # self.img_dir = './data/kaggle_3m/'
#         # self.label_dir = './data/kaggle_3m/'
#         # root = './data/lung/'
#         # root = './data/lung_new/'
#         # root = './data/lung/512/'
#         # self.img_dir = './data/lung/2d_images/'
#         # self.label_dir = './data/lung/2d_masks/'
#         # self.img_dir = './data/lung_new/images/'
#         # self.label_dir = './data/lung_new/masks/'
#         if mode=='train':
#             npz_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
#             cb_npz_file = './data/{}/{}_combine_train_{}_{}_{}.npz'.format(self.data_path,self.data_path,model,type,target_class)
#             self.img_dir = './data/{}/train/imgs/'.format(self.data_path, self.data_path)
#             train_combine = 1
#             if train_combine:
#                 # self.img_dir = './output/adv/{}/train/{}/{}/{}/'.format(self.data_path,'UNet','DAG_A','0')
#                 self.img_dir = './output/adv/{}/train/UNet/ifgsm/1/'.format(self.data_path)
#         if mode=='val':
#             npz_file = './data/{}/{}_val.npz'.format(self.data_path,self.data_path)
#             cb_npz_file = './data/{}/{}_combine_val_{}_{}_{}.npz'.format(self.data_path,self.data_path,model,type,target_class)
#             # self.img_dir = './data/{}/train/imgs/'.format(self.data_path, self.data_path)
#             self.img_dir = './output/adv/{}/val/UNet/ifgsm/1/'.format(self.data_path)
#         elif mode=='test':
#             npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
#             self.img_dir = './data/{}/test/imgs/'.format(self.data_path,self.data_path)
#             # npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
#             # self.img_dir = './data/{}/train/imgs/'.format(self.data_path, self.data_path)
#         # #fundus
#         adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(self.data_path, self.data_path, mode, 'UNet',
#                                                                           'ifgsm',
#                                                                           '1')
#         #octa
#         # adv_npz_file = './data/{}/denoiser/{}_adv_{}_{}_{}_{}.npz'.format(self.data_path, self.data_path, mode, 'UNet',
#         #                                                                   'DAG_A',
#         #                                                                   '0')
#         self.images = []
#         self.labels = []
#
#         ratio = 0.84
#         # data form [images, labels]
#         # with open(self.data_path, 'rb') as fp:
#         #     data = pickle.load(fp)
#
#         # cm = plt.cm.jet
#         # cm = plt.get_cmap('gist_rainbow', 1000)
#         # cm= plt.get_cmap('viridis', 28)
#         # np.random.seed(1)
#         # print(self.img_dir)
#         # #test adv mode
#         if not os.path.exists(cb_npz_file):
#             print('not found existed dump cb', cb_npz_file)
#             if not os.path.exists(adv_npz_file):
#                 print('not found existed dump adv', adv_npz_file)
#                 test_org_data = np.load(npz_file)
#                 self.images+= list(test_org_data['a'])
#                 self.labels = list(test_org_data['b'])*2
#
#                 def filename(x):
#                     return int(x[:-4])
#                 ls_names = sorted(os.listdir(self.img_dir),key=filename)
#                 for img_name in ls_names:
#                     if 'png' in img_name:
#                         img_path = os.path.join(self.img_dir, img_name)
#                         # print('label', label_path)
#                         if channels == 1:
#                             img = cv2.imread(img_path,0)
#                         else:
#                             img = cv2.imread(img_path)
#                         # img = cv2.resize(img,(256,256))/ 255
#                         img = img/255
#                         self.images.append(img)
#                         # print('checklabel', np.unique(data[i][1]))
#                 np.savez_compressed(cb_npz_file, a=self.images, b=self.labels)
#             else:
#                 data = np.load(npz_file)
#                 # term = np.load('/data/linhld/projects/Pytorch_AdversarialAttacks/data/lung_new/color/lung_new_train.npz')
#                 self.images = data['a']
#                 self.labels = data['b']
#                 print('load ', adv_npz_file)
#                 adv_data = np.load(adv_npz_file)
#                 # print('shape', len(adv_data['a']), adv_data['a'].shape)
#                 self.images = np.append(self.images,adv_data['a'], axis = 0)
#                 self.labels =np.repeat(self.labels, 2,axis = 0)
#                 # self.images = self.images
#                 # self.labels = self.images[:591]
#                 np.savez_compressed(cb_npz_file, a=self.images, b=self.labels)
#                 # print('len label', len(self.labels))
#                 # exit()
#         else:
#             print('load combine dataset', cb_npz_file)
#             data = np.load(cb_npz_file)
#             # term = np.load('/data/linhld/projects/Pytorch_AdversarialAttacks/data/lung_new/color/lung_new_train.npz')
#             self.images = data['a']
#             self.labels = data['b']
#         # if 1:
#
#         # print('len', len(self.labels))
#         # exit(0)

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
                cv2.imwrite('{}/{}'.format(out_img_dir, img_name), org_img)
                cv2.imwrite('{}/{}'.format(out_label_dir, img_name), label)
                unique = np.unique(label)
                # print('unique', unique)
                for i, cl in enumerate(unique):
                    label[label == cl] = i
                # print('after', np.unique(label))
                images.append(img)
                labels.append(label)
        return images, labels

# normal
    def __init__(self, root_dir, num_class, channels, mode='train', gen_mode=None, model=None, attack_type=None, data_type='org',
                 mask_type=None, target_class=None, width=256, height=256, suffix=''):
        self.num_classes = num_class
        self.data_path = root_dir
        # train_img_dir = './data/{}/train/imgs/'.format(root_dir)
        # val_img_dir = './data/{}/val/imgs/'.format(root_dir)
        # test_img_dir = './data/{}/test/imgs/'.format(root_dir)
        # train_label_dir = './data/{}/train/labels/'.format(root_dir)
        # val_label_dir = './data/{}/val/labels/'.format(root_dir)
        # test_label_dir = './data/{}/test/labels/'.format(root_dir)
        out_img_dir = './data/{}/{}/imgs/'.format(root_dir, mode)
        out_label_dir = './data/{}/{}/gt/'.format(root_dir, mode)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_label_dir):
            os.makedirs(out_label_dir)
        # if not os.path.exists(train_img_dir):
        #     os.makedirs(train_img_dir)
        # if not os.path.exists(val_img_dir):
        #     os.makedirs(val_img_dir)
        # if not os.path.exists(test_img_dir):
        #     os.makedirs(test_img_dir)
        # if not os.path.exists(train_label_dir):
        #     os.makedirs(train_label_dir)
        # if not os.path.exists(val_label_dir):
        #     os.makedirs(val_label_dir)
        # if not os.path.exists(test_label_dir):
        #     os.makedirs(test_label_dir)
        root = './data/{}/'.format(self.data_path)
        # self.img_dir = './data/eye/{}/imgs/'.format(mode)
        # self.label_dir = './data/eye/{}/gt/'.format(mode)
        if mode == 'adv':
            self.img_dir = './data/{}/{}/imgs/'.format(self.data_path, gen_mode)
            self.label_dir = './data/{}/{}/gt/'.format(self.data_path, gen_mode)
        elif self.data_path!='octafull':
                self.img_dir = './data/{}/{}/imgs/'.format(self.data_path,mode)
                self.label_dir = './data/{}/{}/gt/'.format(self.data_path,mode)
        else:
            self.img_dir = './data/OCTA-500/{}/imgs/'.format(self.data_path)
            self.label_dir = './data/OCTA-500/{}/gt/'.format(self.data_path)
        if mode=='train':
            npz_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
        elif mode == 'val':
            npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
        elif mode=='test':
            if data_type=='org':
                npz_file = './data/{}/{}_test.npz'.format(self.data_path,self.data_path)
                # npz_file = './data/{}/512/{}_test.npz'.format(self.data_path, self.data_path)
            else:
                # if mask_type !='':
                #     npz_file = './data/{}/{}_adv_{}_{}_m{}.npz'.format(self.data_path,self.data_path, model, type,
                #                                             mask_type)
                # if target_class:
                if self.data_path=='brain':
                    npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path,suffix, self.data_path, model, attack_type,
                                                                         mask_type, target_class)
                else:
                    npz_file = './data/{}/{}/{}_adv_{}_m{}t{}.npz'.format(self.data_path, attack_type, self.data_path,
                                                                             data_type,
                                                                             mask_type, target_class)
                # npz_file = './data/{}/{}_adv_{}_mix_label.npz'.format(self.data_path, self.data_path, type)
                # npz_file = './data/{}/{}_adv_{}_pure_target.npz'.format(self.data_path, self.data_path, type)
                # else:
                #     npz_file = './data/{}/{}_adv_{}_{}.npz'.format(self.data_path,suffix, self.data_path, model, type
                #                                                       )
                if not os.path.exists('./data/{}/{}/'.format(self.data_path,attack_type)):
                    os.makedirs('./data/{}/{}/'.format(self.data_path,attack_type))
                npz_org_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # self.img_dir = './output/adv/{}/{}/{}/'.format(self.data_path,model, type)
                # if mask_type !='':
                #     self.img_dir = './output/adv/{}/{}/{}/{}/m{}/'.format(self.data_path, mode, model, type, mask_type)
                # else:
                self.img_dir = './output/adv/{}/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, attack_type, data_type,
                                                                         mask_type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/mix_label/'.format(self.data_path, mode, model, type,
                #                                                          mask_type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/pure_target/'.format(self.data_path, mode, model, type,
                #                                                          mask_type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/'.format('fundus_512', model, type, target_class)
        else:
            npz_file = './data/{}/{}_{}.npz'.format(self.data_path,self.data_path, gen_mode)
            self.img_dir = './data/{}/{}/imgs/'.format(self.data_path, gen_mode)
            # npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/train/imgs/'.format(self.data_path)
            # npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/val/imgs/'.format(self.data_path)
        print('img dir ', self.img_dir)
        print('npz file', npz_file, self.img_dir,  self.label_dir)
        self.images = []
        self.labels = []

        # data form [images, labels]
        # with open(self.data_path, 'rb') as fp:
        #     data = pickle.load(fp)

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
        else:
            #test adv mode
            if not os.path.exists(npz_file):
                test_org_data = np.load(npz_org_file)
                self.labels = test_org_data['b']
                print('not found existed dump adv',npz_file)
                def filename(x):
                    return int(x[:-4])
                ls_names = sorted(os.listdir(self.img_dir),key=filename)
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
                self.images = data['a']
                print(self.images.dtype)
                self.labels = data['b']

class AgDataset(SampleDataset):
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, attack_type, data_type, mask_type, target_class, width,
                 height):
        super(SampleDataset, self).__init__()
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
        # if not os.path.exists(train_img_dir):
        #     os.makedirs(train_img_dir)
        # if not os.path.exists(val_img_dir):
        #     os.makedirs(val_img_dir)
        # if not os.path.exists(test_img_dir):
        #     os.makedirs(test_img_dir)
        # if not os.path.exists(train_label_dir):
        #     os.makedirs(train_label_dir)
        # if not os.path.exists(val_label_dir):
        #     os.makedirs(val_label_dir)
        # if not os.path.exists(test_label_dir):
        #     os.makedirs(test_label_dir)
        # root = './data/OCTA-500/OCTA_3M/'
        root = './data/{}/'.format(self.data_path)
        # root = './data/lung_new/'
        # self.img_dir = './data/lung_new/images/'
        # self.label_dir = './data/lung_new/masks/'
        self.img_dir = './data/eye/{}/imgs/'.format(mode)
        self.label_dir = './data/eye/{}/gt/'.format(mode)
        if mode=='train':
            if data_type == 'org':
                npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            else:
                npz_org_file = './data/{}/{}_train.npz'.format(self.data_path,self.data_path)
                npz_file = './data/{}/denoiser/{}/{}_adv_train_{}_{}_m{}t{}.npz'.format(self.data_path, attack_type, self.data_path,
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
                if self.data_path == 'brain':
                    npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path,attack_type, self.data_path, model, data_type,
                                                                       mask_type, target_class)
                else:
                    npz_file = './data/{}/{}/{}_adv_{}_m{}t{}.npz'.format(self.data_path, attack_type, self.data_path,
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
                if not os.path.exists('./data/{}/{}/'.format(self.data_path,attack_type)):
                    os.makedirs('./data/{}/{}/'.format(self.data_path,attack_type))
                npz_org_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # self.img_dir = './output/adv/{}/{}/{}/'.format(self.data_path,model, type)
                # if mask_type != '':
                # self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type, mask_type, target_class)
                self.img_dir = './output/adv/{}/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, attack_type,\
                    data_type, mask_type, target_class)

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

        # data form [images, labels]
        # with open(self.data_path, 'rb') as fp:
        #     data = pickle.load(fp)

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
                self.labels = data['b']
                # test_org_data = np.load(npz_org_file)
                # self.images = test_org_data['a']
                # self.labels = test_org_data['b']
                # # print('writing')
                # for i, img in enumerate(self.images):
                #     cv2.imwrite('./{}/{}.png'.format(out_img_dir, i), img*255)
                # for i, img in enumerate(self.labels):
                #     cv2.imwrite('./{}/{}.png'.format(out_label_dir, i), img*255)
        print('len imgs', len(self.images))
        if type=='semantic':
            print("norm")
            self.torch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
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