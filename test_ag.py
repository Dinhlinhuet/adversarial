# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import numpy as np
import os
import argparse
import time
from pylab import *
from dataset.dataset import AgDataset
from dataset.scale_att_dataset import AttackDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from util import make_one_hot
from model.AgNet.core.utils import get_model, dice_loss
from opts import get_args

# --------------------------------------------------------------------------------

# # --------------------------------------------------------------------------------
# parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# # ---------------------------
# # params do not need to change
# # ---------------------------
# parser.add_argument('--epochs', type=int, default=250,
#                     help='the epochs of this run')
# parser.add_argument('--n_class', type=int, default=3,
#                     help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
# parser.add_argument('--lr', type=float, default=0.0015,
#                     help='initial learning rate')
# parser.add_argument('--GroupNorm', type=bool, default=True,
#                     help='decide to use the GroupNorm')
# parser.add_argument('--BatchNorm', type=bool, default=False,
#                     help='decide to use the BatchNorm')
# parser.add_argument('--channels', dest='channels', default=3, type=int,
#                       help='number of channels')
# # ---------------------------
# # model
# # ---------------------------
# parser.add_argument('--data_path', type=str, default='../data/',
#                     help='dir of the all img')
# parser.add_argument('--model_path', type=str,  default='final.pth',
#                     help='the pretrain model')
# parser.add_argument('--model_id', type=int, default=0,
#                     help='the id of choice_model in models_list')
# parser.add_argument('--batch_size', type=int, default=1,
#                     help='the num of img in a batch')
# parser.add_argument('--img_size', type=int, default=256,
#                     help='the train img size')
# parser.add_argument('--my_description', type=str, default='test',
#                     help='some description define your train')
# parser.add_argument('--mode', dest='mode', type=str, default='test',
#                   help='mode test origin or adversarial')
# parser.add_argument('--data_type', dest='data_type', type=str, default='',
#                   help='org or DAG')
# parser.add_argument('--attacks', dest='attacks', type=str, default="",
#                       help='attack types: Rician, DAG_A, DAG_B, DAG_C')
# parser.add_argument('--output_path', dest='output_path', type=str,
#                       default='./output/test/', help='output_path')
# parser.add_argument('--adv_model', dest='adv_model', type=str,default='',
#                       help='model name(UNet, SegNet, DenseNet)')
# parser.add_argument('--target', dest='target', default='0', type=str,
#                       help='target class')
# parser.add_argument('--model', dest='model', type=str,default='',
#                       help='model name(UNet, SegNet, DenseNet)')
# parser.add_argument('--mask_type', dest='mask_type', default="", type=str,
#                       help='adv mask')
# # ---------------------------
# # GPU
# # ---------------------------
# parser.add_argument('--gpu', type=bool, default=True,
#                     help='dir of the all ori img')
# parser.add_argument('--gpu_avaiable', type=str, default='0',
#                     help='the gpu used')

args = get_args()


def fast_test(model, args, model_name):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    model = model.double()
    model.eval()
    # Background_IOU = []
    # Vessel_IOU = []
    # ACC = []
    # SE = []
    # SP = []
    # AUC = []
    suffix = args.suffix
    # test_dataset = AgDataset(args.data_path, args.classes, args.channels, args.mode, args.adv_model, args.attacks, args.target,
    #                              args.data_type, args.width, args.height, args.mask_type, suffix)
    test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    args.output_path = os.path.join(args.output_path, args.data_path, args.model, args.mode, args.suffix)
    # if args.mask_type != "":
    # args.output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
    #                                 args.data_type, args.attacks, 'm' + args.mask_type+'t'+args.target, suffix)
    # else:
    #     args.output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
    #                                     args.data_type, args.attacks, args.target)
    print('output path', args.output_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    start = time.time()
    cm = plt.get_cmap('gist_rainbow', 1000)
    dices = []

    for batch_idx, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device).double()
        # labels = labels.to(device).long().squeeze(0)
        labels = labels.to(device).long()

        out, side_5, side_6, side_7, side_8 = model(imgs)
        out = torch.log(softmax_2d(side_8) + EPS)

        # out = F.upsample(out, size=(imgs.size()[2],imgs.size()[3]), mode='bilinear')
        # out = out.cpu().data.numpy()
        # y_pred =out[:,1,:,:]
        # y_pred = y_pred.reshape([-1])
        # print('be', out.size())
        out = torch.argmax(out, 1)
        # print('uot', out.size())
        ppi = make_one_hot(out, args.classes, device)
        # print('ppi', ppi.shape, labels.shape)
        labels = make_one_hot(labels, args.classes, device)
        dice = dice_loss(ppi, labels).detach().cpu().numpy()
        dices.append(dice)
        print('dice score', dice)
        # meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)
        # Auc = roc_auc_score(tmp_gt, y_pred)
        # AUC.append(Auc)

        # Background_IOU.append(IU[0])
        # Vessel_IOU.append(IU[1])
        # ACC.append(Acc)
        # SE.append(Se)
        # SP.append(Sp)
        # print('onehotcvt', masks.shape)
        for i, mask in enumerate(out.detach().cpu().numpy()):
            # print(np.unique(mask))
            # mask=mask/255
            mask = mask / args.classes
            output = np.uint8(cm(mask) * 255)
            # output = np.uint8(mask * 255)
            # output = np.uint8(cm(mask) * args.classes)
            # print(np.unique(cm(mask)))
            # print('min,ma', np.min(output), np.max(output), output.shape)
            output = Image.fromarray(output)
            output.save(os.path.join(args.output_path, '{}.png'.format(batch_idx * args.batch_size + i)))
    end = time.time()

        # print(str(i+1)+r'/'+str(len(img_list))+': '+'| Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f} | Auc: {:.3f} |  Background_IOU: {:f}, vessel_IOU: {:f}'.format(Acc,Se,Sp,Auc,IU[0], IU[1])+'  |  time:%s'%(end-start))

    # print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
    print(str('val: {:s} |  dice: {:.3f}').format(model_name, 1-np.mean(dices)))

    # store test information
    # with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'a+') as f:
    #     f.write('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
    #     f.write('\n\n')

    # return np.mean(np.stack(Vessel_IOU))


if __name__ == '__main__':
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable
    gpu = args.gpus
    device_ids = []
    if gpu:

        if not torch.cuda.is_available():
            print("No cuda available")
            raise SystemExit

        device = torch.device(0)

    else:
        device = torch.device("cpu")

    model_name = 'AG_Net'

    model = get_model(model_name)
    model = model(n_classes=args.classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
    model = model.to(device)

    if args.gpus>1:
        model.cuda()
        # model_path = '../models/AG_Net.pth'
    print('model', args.model_path, args.data_path, args.model)
    model_path = os.path.join(args.model_path, args.data_path, args.model+ '.pth')
    model.load_state_dict(torch.load(model_path))
    print('success load models: %s' % (model_name))

    print ('This model is %s_%s' % (model_name, args.classes))

    # test_img_list = get_img_list(args.data_path, flag='test')

    fast_test(model, args, model_name)



