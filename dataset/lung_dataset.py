import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
#from scipy.misc import cv2.imread, imresize
import cv2
import re
import glob

def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

class BaseDataset:
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, attack_type, data_type,
                 mask_type=None, target_class=None, width=255,height=255,suffix=None):
        self.num_classes = num_class
        self.data_path = root_dir
        out_img_dir = './data/{}/{}/img/'.format(root_dir, mode)
        out_label_dir = './data/{}/{}/gt/'.format(root_dir, mode)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_label_dir):
            os.makedirs(out_label_dir)
        print('mode', mode)
        root = './data/{}/'.format(self.data_path)
        self.img_dir = './data/lung/imgs/{}'.format(mode)
        self.label_dir = './data/lung/gts/{}'.format(mode)
        print()
        if mode == 'train':
            if data_type == 'org':
                npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            else:
                npz_org_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
                npz_file = './data/{}/denoiser/{}/{}_adv_train_{}_{}_m{}t{}.npz'.format(self.data_path, suffix,
                                                                                        self.data_path,
                                                                                        model,
                                                                                        data_type,
                                                                                        mask_type, target_class)
                self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type,
                                                                         mask_type,
                                                                         target_class)
        elif mode == 'val':
            npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
        elif mode == 'test':
            if data_type == 'org':
                npz_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # npz_file = './data/{}/512/{}_test.npz'.format(self.data_path, self.data_path)
            else:
                npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path, attack_type, self.data_path,
                                                                             model, data_type,
                                                                             mask_type, target_class)
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
                if not os.path.exists('./data/{}/{}/'.format(self.data_path, attack_type)):
                    os.makedirs('./data/{}/{}/'.format(self.data_path, attack_type))
                npz_org_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
                # self.img_dir = './output/adv/{}/{}/{}/'.format(self.data_path,model, type)
                # self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type,
                #                                                          mask_type, target_class)
                self.img_dir = './output/adv/{}/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, attack_type,
                                                data_type, mask_type, target_class)

                # else:
                #     self.img_dir = './output/adv/{}/{}/{}/{}/'.format(self.data_path, model, type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/'.format('fundus_512', model, type, target_class)
        else:
            npz_file = './data/{}/{}_test.npz'.format(self.data_path, self.data_path)
            self.img_dir = './data/{}/test/imgs/'.format(self.data_path)
            # npz_file = './data/{}/{}_train.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/train/imgs/'.format(self.data_path)
            # npz_file = './data/{}/{}_val.npz'.format(self.data_path, self.data_path)
            # self.img_dir = './data/{}/val/imgs/'.format(self.data_path)
        self.images = []
        self.labels = []

        print('img dir', self.img_dir)
        if data_type == 'org':
            # train and test org, generate adv
            if not os.path.exists(npz_file):
                print('not found existed dump', npz_file)
                self.images, self.labels = read_files(self.img_dir, self.label_dir, out_img_dir, out_label_dir,
                                                      channels, width, height)
                images = np.array(self.images)
                labels = np.array(self.labels)
                np.savez_compressed('{}/{}_{}.npz'.format(root, self.data_path, mode), a=images, b=labels)
            else:
                print('load ', npz_file)
                data = np.load(npz_file, allow_pickle=True)
                self.images = data['a']
                self.labels = data['b']
                # print('checklabel', np.unique(self.labels[0]))
                # print('writing')
                # for i, img in enumerate(self.images):
                #     cv2.imwrite('./{}/{}.png'.format(out_img_dir, i), img*255)
                # for i, img in enumerate(self.labels):
                #     cv2.imwrite('./{}/{}.png'.format(out_label_dir, i), img*255)

        else:
            # test adv mode
            if not os.path.exists(npz_file):
                test_org_data = np.load(npz_org_file)
                self.labels = test_org_data['b']
                print('not found existed dump adv', npz_file)
                ls_names = sorted(os.listdir(self.img_dir), key=filename)
                print('ls name', len(ls_names))
                for img_name in ls_names:
                    if 'png' or 'bmp' in img_name:
                        img_path = os.path.join(self.img_dir, img_name)
                        # print('label', label_path)
                        if channels == 1:
                            img = cv2.imread(img_path, 0)
                        else:
                            img = cv2.imread(img_path)
                        img = cv2.resize(img, (256, 256)) / 255
                        # img = img/255
                        self.images.append(img)
                        # print('checklabel', np.unique(data[i][1]))
                np.savez_compressed(npz_file, a=self.images, b=self.labels)
            else:
                print('load ', npz_file)
                data = np.load(npz_file)
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
        # if semantic generate
        # self.torch_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])
        # else:
        self.torch_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pass

class CovidDataset(BaseDataset):
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, attack_type, data_type,
                 mask_type=None, target_class=None, suffix=None, width=255, height=255):
        super(CovidDataset, self).__init__(root_dir, num_class, channels, mode, model, attack_type, data_type,
                    mask_type=mask_type,target_class=target_class, width=width, height=height, suffix=suffix)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        # print('uni', np.unique(labels))
        # print('img', image.shape)

        image = self.torch_transform(image)
        # labels = torch.from_numpy(labels).unsqueeze(0)
        # labels = torch.from_numpy(labels)
        labels = self.torch_transform(labels)
        # print('label', labels.size())
        # print('torh', torch.unique(labels).cpu().numpy())
        return image, labels

class CovidAgDataset(BaseDataset):
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, attack_type, data_type,
                 mask_type=None, target_class=None,width=255, height=255, suffix=None):
        super(CovidAgDataset, self).__init__(root_dir, num_class, channels, mode, model, attack_type, data_type,
                    mask_type=mask_type,target_class=target_class, width=width, height=height, suffix=suffix)

    def __getitem__(self, index):
        # print("idx", index)
        image = self.images[index]
        labels = self.labels[index]
        # print('uni', np.unique(labels))
        # print('img', image.shape)

        image = self.torch_transform(image)
        labels = self.torch_transform(labels)
        # print('label', labels.size())
        labels = labels.squeeze(0)
        # print('label1', labels.size())
        # print('torh', torch.unique(labels).cpu().numpy())
        return image, labels

def read_files(img_dir, label_dir, out_img_dir, out_label_dir, channels, width, height):
    images = []
    labels = []
    idx = 0
    for folder in os.listdir(img_dir):
        print('folder', folder)
        img_sub_dir = os.path.join(img_dir,folder)
        label_sub_dir = os.path.join(label_dir,folder)
        ls_names = os.listdir(img_sub_dir)
        for img_name in ls_names:
            if 'png' or 'bmp' or 'jpg' in img_name:
                # print(self.img_dir,img_name[:-9]+'.png')
                print(img_name)
                img_path = os.path.join(img_sub_dir, img_name)
                if '009' in folder:
                    label_path = os.path.join(label_sub_dir,
                                              img_name.replace('case','coronacases_'))
                else:
                    if 'coro' in img_name:
                        label_path = os.path.join(label_sub_dir,
                                                  img_name.replace('_org', '')).replace('_case','_')
                    else:
                        label_path = os.path.join(label_sub_dir, img_name.replace('org', '').replace('_covid-19-pneumonia-','')).\
                replace('-dcm','')
                print('label', label_path)
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
                label[label < 150] = 0
                label[label > 150] = 255
                label = cv2.resize(label, (height, width), interpolation=cv2.INTER_NEAREST)
                norm_label = label/255
                print('idx', idx)
                cv2.imwrite('{}/{}.png'.format(out_img_dir, idx), org_img)
                cv2.imwrite('{}/{}.png'.format(out_label_dir, idx), label)
                unique = np.unique(label)
                # print('unique', unique)
                for i, cl in enumerate(unique):
                    label[label == cl] = i
                # print('after', np.unique(label))
                images.append(img)
                labels.append(norm_label)
                idx+=1
    return images, labels