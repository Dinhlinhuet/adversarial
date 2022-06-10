import torch
import torch.nn as nn
import numpy as np

import os
import argparse
import time
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from model.AgNet.core.utils import  get_model, dice_loss
from pylab import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from dataset.dataset import SampleDataset, AgDataset
from dataset.lung_dataset import CovidAgDataset
from util import make_one_hot

# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=100,
                    help='the epochs of this run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                  help='manual epoch number (useful on restarts)')
parser.add_argument('--n_class', type=int, default=3,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--channels', type=int, default=3,
                    help='channels')
parser.add_argument('--lr', type=double, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='../data/fundus/test/',
                    help='dir of the all img')
parser.add_argument('--save-dir', default='./checkpoints/', type=str,
                  help='directory to save checkpoint (default: none)')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=256,
                    help='the train img size')
parser.add_argument('--width', type=int, default=256,
                    help='the train img size')
parser.add_argument('--height', type=int, default=256,
                    help='the train img size')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--device', type=int, default=0,
                    help='device ')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------

device = torch.device(args.device)
model_name = 'Ag_Net'
model = get_model(model_name)

model = model(n_classes=args.n_class, n_channels=args.channels, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
model = model.to(device)

n_class = args.n_class
EPS = 1e-12
# define path
data_path = args.data_path
if 'lung' in data_path:
    train_dataset = CovidAgDataset(data_path, n_class, args.channels, mode='train', model=None,
                                 data_type='org', width=args.width, height=args.height)
    val_dataset = CovidAgDataset(data_path, n_class, args.channels, mode='val', model=None,
                               data_type='org', width=args.width, height=args.height)
    train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
    val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
elif 'fundus' in data_path:
    train_dataset = AgDataset(data_path, n_class, 3, 'train',None,None,None,'org', args.img_size, args.img_size)
    val_dataset = AgDataset(data_path, n_class,3, 'val', None, None, None, 'org', args.img_size, args.img_size)
    train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
    val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
else:
    train_dataset = AgDataset(data_path, n_class, 3, 'train', None, None, None, 'org', args.img_size, args.img_size)
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

print('total train image : {}'.format(len(train_sampler)))
print('total val image : {}'.format(len(val_sampler)))

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    sampler=train_sampler,
    drop_last= True
)
if 'lung' in data_path or 'fundus' in data_path:
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=val_sampler,
        drop_last=True
    )
else:
    val_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=val_sampler,
        drop_last=True
    )


if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
save_dir = args.save_dir
model_folder = os.path.abspath('./{}/{}/'.format(save_dir, args.data_path))
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = os.path.join(model_folder, 'AgNet.pth')
# img_list = get_img_list(args.data_path, flag='train')
# test_img_list = get_img_list(args.data_path, flag='test')

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
model = model.double()
criterion = nn.NLLLoss()
softmax_2d = nn.Softmax(dim = 1)

IOU_best = 0

# with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'w+') as f:
#     f.write('This model is %s_%s: ' % (model_name, args.my_description)+'\n')
#     f.write('args: '+str(args)+'\n')
#     f.write('train lens: '+str(len(img_list))+' | test lens: '+str(len(test_img_list)))
#     f.write('\n\n---------------------------------------------\n\n')

start_epoch = args.start_epoch
num_epochs = args.epochs
best_dice = 1e10
for epoch in range(start_epoch,num_epochs+start_epoch):
    model.train()

    begin_time = time.time()
    print ('Epoch %s' % (epoch))
    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 100:
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dices = []
    for i, (images, masks) in enumerate(train_loader):
        # print('img', images.size())
        images = images.to(device).double()
        # tmp_gt = masks
        masks = masks.to(device).long()
        # tmp_gt = masks
        # img, masks, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()
        out, side_5, side_6, side_7, side_8 = model(images)
        # print('out', out.size(), masks.size())
        out = torch.log(softmax_2d(out) + EPS)
        # print('out', out.size(), masks.shape)
        loss = criterion(out, masks)
        loss += criterion(torch.log(softmax_2d(side_5) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_6) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_7) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_8) + EPS), masks)
        out = torch.log(softmax_2d(side_8) + EPS)

        loss.backward()
        optimizer.step()

        ppi = torch.argmax(out, 1)
        ppi =  make_one_hot(ppi, n_class,device)
        # tmp_out = ppi.reshape([-1])
        # print('tm', tmp_out.size())
        # tmp_gt = tmp_gt.reshape([-1])

        # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.double32)
        # meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)
        masks= make_one_hot(masks, n_class, device)
        # print('out', out.size(), masks.size())
        # print('max', torch.min(y_pred), torch.max(y_pred))
        dice = dice_loss(ppi, masks).detach().cpu().numpy()
        print('dice', dice)
        dices.append(dice)
        # print('done')
    # print(str('train: {:s} | epoch_batch: {:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
    #           '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, epoch, loss.item(), Acc, Se, Sp,
    #                                                              IU[0], IU[1]))
    print(str('train: {:s} | epoch_batch: {:d} | loss: {:f}  | dice: {:.3f}').format(model_name, epoch, loss.item(),
                                                                                     np.mean(dices)))
    dices = []
    for i, (images, masks) in enumerate(val_loader):
        # print('img', images.size())
        images = images.to(device).double()
        # tmp_gt = masks
        masks = masks.to(device).long()
        # tmp_gt = masks
        # img, masks, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()

        out, side_5, side_6, side_7, side_8 = model(images)
        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, masks)
        loss += criterion(torch.log(softmax_2d(side_5) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_6) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_7) + EPS), masks)
        loss += criterion(torch.log(softmax_2d(side_8) + EPS), masks)
        out = torch.log(softmax_2d(side_8) + EPS)

        loss.backward()
        optimizer.step()

        ppi = torch.argmax(out, 1)
        ppi = make_one_hot(ppi, n_class, device)
        # tmp_out = ppi.reshape([-1])
        # tmp_gt = tmp_gt.reshape([-1])

        # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.double32)
        # meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)
        # dice = calculate_Accuracy(my_confusion)
        masks = make_one_hot(masks, n_class, device)
        dice = dice_loss(ppi, masks).detach().cpu().numpy()
        dices.append(dice)
    total_dice = np.mean(dices)
    print(str('val: {:s} | epoch_batch: {:d} | loss: {:f}  | dice: {:.3f}').format(model_name, epoch, loss.item(),
                                                                                   total_dice))
    # print(str('val: {:s} | epoch_batch: {:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
    #           '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name,epoch, loss.item(), Acc,Se,Sp,
    #                                                                           IU[0], IU[1]))

    print('training finish, time: %.1f s' % (time.time() - begin_time))

    if total_dice<best_dice:
        best_dice = total_dice
        torch.save(model.state_dict(),model_path)
        print('success save Nucleus_best model')
































