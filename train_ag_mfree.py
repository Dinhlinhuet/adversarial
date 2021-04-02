import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

import os
import argparse
import time
from model.AgNet.core.utils import calculate_Accuracy, get_model, dice_loss
from pylab import *
import random
from torch.utils.data import DataLoader,Sampler
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from dag_utils import generate_target_bs
from dataset.dataset import SampleDataset, AgDataset
from util import make_one_hot
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

# --------------------------------------------------------------------------------

models_list = ['AG_Net']

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=50,
                    help='the epochs of this run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                  help='manual epoch number (useful on restarts)')
parser.add_argument('--n_class', type=int, default=3,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0015,
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
parser.add_argument('--save-dir', default='./checkpoints/combine/', type=str,
                  help='directory to save checkpoint (default: none)')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=256,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='test8',
                    help='some description define your train')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='1',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------

gpu = args.gpu
device_ids = []
if gpu:

    if not torch.cuda.is_available():
        print("No cuda available")
        raise SystemExit

    device = torch.device(0)

else:
    device = torch.device("cpu")
model_name = models_list[args.model_id]
model = get_model(model_name)

model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
model = model.to(device)

# if args.gpu:
#     model.cuda()
#     print('GPUs used: (%s)' % args.gpu_avaiable)
#     print('------- success use GPU --------')

n_class = args.n_class
EPS = 1e-12
# define path
data_path = args.data_path
train_dataset = AgDataset(data_path, n_class, 3, 'train',None,None,None,'org', args.img_size, args.img_size)
val_dataset = AgDataset(data_path, n_class,3, 'val', None,None,None, 'org', args.img_size, args.img_size)
print('total train image : {}'.format(len(train_dataset)))
print('total val image : {}'.format(len(val_dataset)))
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)
# train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
# train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
# val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
train_sampler = SequentialSampler(np.arange(len(train_dataset)))
val_sampler = SequentialSampler(np.arange(len(val_dataset)))

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    sampler=train_sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=4,
    sampler=val_sampler
)

if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
save_dir = args.save_dir
model_folder = os.path.abspath('./{}/{}/'.format(save_dir, args.data_path))
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = os.path.join(model_folder, 'AgNet_{}.pth'.format('advcw'))
# img_list = get_img_list(args.data_path, flag='train')
# test_img_list = get_img_list(args.data_path, flag='test')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
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
len_train = len(train_loader)
best_dice = 1e10
optimizers = []
learning_rate = 0.01
ws = torch.empty([0])
model = model.double()
for i, (images, masks) in enumerate(train_loader):
    w = Variable(torch.atan(images.clone()*2-1), requires_grad=True)
    # w = torch.zeros_like(image, requires_grad=True).to(device)
    # w = torch.nn.Parameter(image, requires_grad=True)
    optimi = optim.Adam([w], lr=learning_rate)
    optimizers.append(optimi)
    w = w.unsqueeze(0)
    ws = torch.cat((ws, w))
ws.to(device)
print('ws', ws.size())
c= 1
b= 1e-1
m = 50
for epoch in range(start_epoch,num_epochs+start_epoch):
    model.train()

    begin_time = time.time()
    print ('Epoch %s' % (epoch))
    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 100:
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dices = []
    train_iterator = iter(train_loader)
    qbar = tqdm(range(len(train_loader)))
    for i in qbar:
        org_images, masks= next(train_iterator)
        masks = masks.to(device).long()
        target_class = random.randint(0, n_class - 1)
        adv_target = generate_target_bs(i, masks, target_class=target_class)
        # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
        # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
        adv_target_oh = make_one_hot(adv_target, n_class, device)
        masks_oh = make_one_hot(masks, n_class, device)
        masks = masks.long()
        org_images = org_images.to(device).double()
        sub_dices = []
        pbar = tqdm(range(m), leave=False)
        for k in pbar:
            images = images.to(device).double()
            images = 1 / 2 * (nn.Tanh()(ws[i]) + 1)
            images = images.to(device).double()
            # print('img', images.size(), images.type())
            # tmp_gt = masks
            # adv_target = adv_target.long()
            #     clean_dice = ag_dice_loss(predictions, ground_truth)
            # tmp_gt = masks
            # img, masks, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
            optimizer.zero_grad()
            optimizers[i].zero_grad()
            # print('img', images.size(), images.type())
            out, side_5, side_6, side_7, side_8 = model(images)
            # print('out', out.size(), masks.size())
            out = torch.log(softmax_2d(out) + EPS)
            # print('out', out.size())
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
            adv_dice = dice_loss(ppi, adv_target_oh)
            clean_dice = dice_loss(ppi, masks_oh)
            adv_direction = adv_dice-clean_dice
            mse = nn.MSELoss(reduction='sum')(images, org_images)
            cost = mse*b + adv_direction
            cost = adv_direction
            cost.requires_grad_()
            cost.backward()
            optimizers[i].step()
            # tmp_out = ppi.reshape([-1])
            # print('tm', tmp_out.size())
            # tmp_gt = tmp_gt.reshape([-1])
    
            # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
            # meanIU, Acc,Se,Sp,IU = calculate_Accuracy(my_confusion)
            # print('out', out.size(), masks.size())
            # print('max', torch.min(y_pred), torch.max(y_pred))
            # dice = dice_loss(ppi, masks).detach().cpu().numpy()
            # print('dice', adv_dice)
            # print('dice', clean_dice)
            clean_dice = clean_dice.detach().cpu().numpy()
            sub_dices.append(clean_dice)
            pbar.set_description('%2.2f clean dice %f' % (((k + 1) / m * 100),clean_dice))
        qbar.set_description('%2.2f dice %f' % (((i + 1) /len_train  * 100), np.mean(sub_dices)))
        # print('sub_dice', np.mean(sub_dices))
        dices+=sub_dices
            # print('done')
    # print(str('train: {:s} | epoch_batch: {:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
    #           '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, epoch, loss.item(), Acc, Se, Sp,
    #                                                              IU[0], IU[1]))
    print(str('train: {:s} | epoch_batch: {:d} | loss: {:f}  | dice: {:.3f}').format(model_name, epoch, loss.item(),
                                                                                     np.mean(dices)))
    dices = []
    for i, (images, masks) in enumerate(val_loader):
        # print('img', images.size())
        images = images.to(device).float()
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

        # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
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
































