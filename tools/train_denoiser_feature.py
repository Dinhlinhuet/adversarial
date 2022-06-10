import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from torchsummary import summary
from collections import defaultdict
import copy
from os import path
from optparse import OptionParser
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from loss import onehot2norm, dice_loss, make_one_hot

from data import DefenseDataset, DefenseSclDataset
from dataset.semantic_dataset import DefenseSemanticDataset
from model.UNet_Nonlocal import UNetNonlocal
from model.DenseNet_Nonlocal import DenseNetNonlocal
from model.SegNet_Nonlocal import SegNetNonlocal
from model.AgNet.core.AgNet_Nonlocal import AG_NetNonlocal
from model import deeplab
from util import save_metrics, print_metrics
from loss import combined_loss


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('--classes', dest='classes', default=2, type='int',
                      help='number of classes')
    parser.add_option('--batch_size', dest='batch_size', default=2, type='int',
                      help='batch size')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--GroupNorm', action="store_true", default= True,
                      help='decide to use the GroupNorm')
    parser.add_option('--BatchNorm', action="store_false", default=False,
                        help='decide to use the BatchNorm')
    parser.add_option("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_option("--output_stride", type='choice', default=16, choices=[8, 16])
    parser.add_option('--suffix', dest='suffix', type='string',
                      default='', help='suffix to purpose')
    parser.add_option('--data_type', dest='data_type', type='string',
                      default='', help='data type to purpose')
    parser.add_option('--resume', default='', type='string', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_option('--target_model', default='./checkpoints/', type='string', metavar='PATH',
                      help='path to target model (default: none)')
    parser.add_option('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_option('--save-dir', default='./checkpoints/denoiser/', type='string', metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_option('--device', dest='device', default=0, type='int',
                      help='device index number')

    (options, args) = parser.parse_args()
    return options

def train_net(model, args, denoiser_path):
    data_path = args.data_path
    num_epochs = args.epochs
    n_classes = args.classes
    print('model name', args.model)
    # set device configuration
    device = torch.device(args.device)

    params = model.parameters()
    # torch.cuda.set_device(gpu)
    # denoiser.cuda(gpu)
    # pytorch_total_params = sum(p.numel() for p in c_params)
    # print('total denoiser params', pytorch_total_params)
    optimizer = torch.optim.Adam(params)
    model = model.to(device)
    # model.cuda(gpu)

    # set image into training and validation dataset
    val_dataset = None
    val_sampler = None
    if 'scl' in args.data_type:
        train_dataset = DefenseSclDataset(data_path, 'train', args.channels)
    elif any(data_name in args.data_type for data_name in ['dag','ifgsm']):
        train_dataset = DefenseDataset(data_path,'train', args.channels, args.data_type)
    else:
        train_dataset = DefenseSemanticDataset(data_path, 'train', args.channels, args.data_type)

    if any(data_name in data_path for data_name in ['fundus', 'lung']):
        if any(data_name in args.data_type for data_name in ['dag', 'ifgsm']):
            if 'AgNet' != args.model:
                val_dataset = DefenseDataset(data_path, 'val', args.channels, args.data_type)
                val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
        else:
            val_dataset = DefenseSemanticDataset(data_path, 'val', args.channels, args.data_type)
            val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
        train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
    else:
        train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
    print('total image : {}'.format(len(train_dataset)*2))

    # train_size = len(train_dataset)*2
    # val_size = len(val_dataset)*2
    train_size = len(train_sampler) * 2
    print('train',train_size)
    if 'AgNet'!=args.model:
        val_size = len(val_sampler) * 2
        print('val',val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=train_sampler
    )
    if 'AgNet' != args.data_path:
        if any(data_name in data_path for data_name in ['fundus', 'lung']):
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                sampler=val_sampler
            )
        else:
            val_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                sampler=val_sampler
            )
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        print('not resume')
    print('save path', denoiser_path)
    # set optimizer


    best_dice = 1e10
    best_along_train_dice = 1e10
    best_cb = 1e10
    best_along_train_cb = 1e10

    ## for early stopping 
    early_stop = False
    patience = 7
    counter = 0
    for epoch in range(start_epoch,num_epochs+start_epoch):
        print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
        model.train()
        metrics = defaultdict(float)
        epoch_size = 0
        # train model
        for batch_idx, (images, adv_images, masks) in enumerate(train_loader):
            # print('img', adv_images.size())
            adv_images = adv_images.to(device)
            if args.model=='AgNet':
                # print('here')
                adv_images = adv_images.double()
            else:
                adv_images = adv_images.float()
            masks = masks.to(device).long()

            # images = images.cuda(non_blocking=True).float()
            # adv_images = adv_images.cuda(non_blocking=True).float()
            # masks = masks.cuda(non_blocking=True).long()

            optimizer.zero_grad()
            if args.model=='AgNet':
                out, side_5, side_6, side_7, side_8 = model(adv_images)
                # print('out', out.size(), masks.size())
                # out = torch.log(softmax_2d(out) + EPS)
            else:
                out = model(adv_images)
            cb_loss, cross, dice = combined_loss(out, masks, device, n_classes)
            # print('mask', masks.squeeze(1).size(), outputs.size())
            # print('clean', dice.mean())
            # print('adv', adv_dice.mean())
            save_metrics(metrics, adv_images.size(0), cb_loss, cross, dice)
            cb_loss.backward()
            optimizer.step()

            # statistics
            epoch_size += images.size(0)

            del images, masks
        train_cb_loss = metrics['loss'] / train_size
        train_dice = metrics['dice'] / train_size
        print('combine loss: {:.4f} , dice {:.4f}'.format(
            train_cb_loss, train_dice))
        # values = print_metrics(metrics, epoch_size, 'train')
        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))
        if args.model!='AgNet':
            model.eval()
            metrics = defaultdict(float)
            # validate model
            for images, adv_images, masks in val_loader:
                adv_images = adv_images.to(device)
                masks = masks.to(device).long()
                # print('adv', adv_images.size())
                if args.model == 'AgNet':
                    # print('here')
                    adv_images = adv_images.double()
                else:
                    adv_images = adv_images.float()

                if args.model=='AgNet':
                    out, side_5, side_6, side_7, side_8 = model(adv_images)
                    # print('out', out.size(), masks.size())
                    # out = torch.log(softmax_2d(out) + EPS)
                else:
                    out = model(adv_images)
                cb_loss, cross, dice = combined_loss(out, masks.squeeze(1), device, n_classes)
                save_metrics(metrics, adv_images.size(0), cb_loss, cross, dice)

                # statistics
                epoch_size += images.size(0)

                # del images, masks, outputs, loss, cross, dice
                del images, masks
            val_cb_loss = metrics['loss'] / val_size
            val_dice = metrics['dice'] / val_size
        else:
            val_cb_loss = train_cb_loss
            val_dice = train_dice
        print('Adv combine loss: {:.4f} , dice {:.4f}'.format(val_cb_loss, val_dice))
        # print_metrics(metrics, epoch_size, 'val')
        #
        # save model if best validation loss
        if val_cb_loss < best_cb:
            print("saving best model")
            #dice
            best_cb = val_cb_loss
            best_dice = val_dice
            best_along_train_dice = train_dice
            #cb
            best_along_train_cb = train_cb_loss

            model_copy = copy.deepcopy(model)
            model_copy = model_copy.cpu()

            model_state_dict = model_copy.state_dict()
            torch.save(model_state_dict, denoiser_path)

            del model_copy

            counter = 0

        else:
            counter += 1
            print('EarlyStopping counter : {:>3} / {:>3}'.format(counter, patience))

            if counter >= patience :
                early_stop = True

        print('dice val loss {:4f} dice train loss {:4f}||'
              'combine val loss {:4f} combine train loss {:4f}'
            .format(best_dice, best_along_train_dice,
                    best_cb, best_along_train_cb))

        if early_stop :
            print('Early Stopping')
            break
        
    return
    
    
if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    target_dir = './checkpoints/'
    if args.model == 'UNet' or 'SegNet' or 'DenseNet' or 'AgNet':
        args.target_model = '{}/{}/{}.pth'.format(target_dir, args.data_path, args.model)
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit
    if args.model == 'UNet':
        model = UNetNonlocal(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'SegNet':
        model = SegNetNonlocal(in_channels=n_channels, n_classes=n_classes)
    elif args.model == 'AgNet':
        model = AG_NetNonlocal(n_classes=n_classes, n_channels=args.channels, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        # model = get_model('AG_Net')
        # model = model(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        model = model.double()
    elif args.model == 'DenseNet':
        model = DenseNetNonlocal(in_channels=n_channels, n_classes=n_classes)
    elif 'deeplab' in args.model:
        model = deeplab.modeling_nonlocal.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride,
                                                      in_channels=args.channels, pretrained_backbone=False)
        if args.separable_conv and 'plus' in args.model:
            deeplab.convert_to_separable_conv(model.classifier)

    denoiser_folder = os.path.join(args.save_dir, args.data_path, args.data_type)
    if not os.path.exists(denoiser_folder):
        os.makedirs(denoiser_folder)
    denoiser_path = os.path.join(denoiser_folder, '{}_{}.pth'.format(args.model,args.suffix))
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    train_net(model, args, denoiser_path)