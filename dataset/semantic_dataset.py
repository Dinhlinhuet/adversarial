import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
#from scipy.misc import cv2.imread, imresize
import cv2
import time
import re

def filename(x):
    return int(re.sub('[^0-9]','', x.split('.')[0]))

class DefenseSemanticDataset(Dataset):
    def __init__(self, data_path, phase, channels, data_type=None):
        assert phase == 'train' or phase == 'val' or phase == 'test'
        self.phase = phase
        # self.dataset = dataset
        # self.attack = attack
        self.data_path = data_path
        npz_file = './data/{}/{}_{}.npz'.format(data_path,data_path, phase)

        adv_data_dir = './data/{}/denoiser/'.format(data_path)
        if not os.path.exists(adv_data_dir):
            os.makedirs(adv_data_dir)
        if data_path=='brain':
            attk_model = 'deeplabv3plus_mobilenet' #brain
        elif data_path=='fundus':
            attk_model = 'SegNet' #fundus
        adv_npz_file = '{}/semantic_attk_{}_{}.npz'.format(adv_data_dir, data_path, phase)
        adv_dir = './output/adv/{}/{}/{}/semantic/m5t0/'.format(data_path, phase, attk_model)
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
        debug = False
        if debug:
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

class DefenseSemanticTestDataset(Dataset):
    def __init__(self, data_path,phase, channels, model, data_type, target_class,mask_type):
        assert phase == 'train' or phase == 'val' or phase == 'test'
        self.phase = phase
        # self.dataset = dataset
        # self.attack = attack
        self.data_path = data_path
        npz_file = './data/{}/{}_{}.npz'.format(data_path,data_path, phase)

        # adv_npz_file = './data/{}/denoiser/scl_attk_{}_{}.npz'.format(data_path, data_path, phase)
        if data_path=='brain':
            adv_npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path, data_type, self.data_path, model,
                                                                 data_type,
                                                                 mask_type, target_class)
        else:
            adv_npz_file = './data/{}/{}/{}_adv_{}_m{}t{}.npz'.format(self.data_path, data_type, self.data_path,
                                                                         data_type, mask_type, target_class)
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

class SampleDataset:
    # normal
    def __init__(self, root_dir, num_class, channels, mode, model, type, target_class, data_type, width,
                 height,mask_type, suffix):
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
                if self.data_path == 'brain':
                    npz_file = './data/{}/{}/{}_adv_{}_{}_m{}t{}.npz'.format(self.data_path,suffix, self.data_path, model, data_type,
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
                self.img_dir = './output/adv/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, data_type, mask_type, target_class)
                # self.img_dir = './output/adv/{}/{}/{}/{}/{}/m{}t{}/'.format(self.data_path, mode, model, type, data_type, mask_type,
                #                                                          target_class)

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
        #if semantic generate
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