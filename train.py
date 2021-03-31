import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,Sampler
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchsummary import summary
from collections import defaultdict
import copy
import pickle
import time

from optparse import OptionParser

from dataset.dataset import SampleDataset
from model import UNet, SegNet, DenseNet
from util import save_metrics, print_metrics
from loss import combined_loss

def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('--classes', dest='classes', default=2, type='int',
                      help='number of classes')
    parser.add_option('--batch_size', dest='batch_size', default=6, type='int',
                      help='batch size')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--resume', default='', type='string', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_option('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_option('--save-dir', default='./checkpoints/', type='string', metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_option('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_option('--device1', dest='device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')

    (options, args) = parser.parse_args()
    return options

def train_net(model, args):
    
    data_path = args.data_path
    num_epochs = args.epochs
    gpu = args.gpu
    n_classes = args.classes
    data_width = args.width
    data_height = args.height

    # set device configuration
    device_ids = []

    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.device1)
        
        device_ids.append(args.device1)
        
        if args.device2 != -1 :
            device_ids.append(args.device2)
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)
        
    
    else :
        device = torch.device("cpu")
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids = device_ids)

    model = model.to(device)

    # set image into training and validation dataset
    
    train_dataset = SampleDataset(data_path, n_classes, n_channels, 'train',None,None,None,'org', args.width, args.height)
    val_dataset = SampleDataset(data_path, n_classes, n_channels, 'val', None, None, None, 'org', args.width, args.height)
    print('total train image : {}'.format(len(train_dataset)))
    print('total val image : {}'.format(len(val_dataset)))
    # train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
    val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=val_sampler
    )

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        print('not resume')
        # if start_epoch == 0:
        #     start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join(args.save_dir, exp_id)
        else:
            save_dir = args.save_dir
    print('save dir', save_dir)
    # model_folder = os.path.abspath('./checkpoints/{}/'.format(args.data_path))
    model_folder = os.path.abspath('./{}/{}/'.format(save_dir,args.data_path))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    if args.model == 'UNet':
        model_path = os.path.join(model_folder, 'UNet.pth')
    
    elif args.model == 'SegNet':
        model_path = os.path.join(model_folder, 'SegNet.pth')
        
    elif args.model == 'DenseNet':
        model_path = os.path.join(model_folder, 'DenseNet.pth')
    print('save path', model_path)
    # set optimizer
    
    optimizer = torch.optim.Adam(model.parameters())


    # main train
    
    display_steps = 30
    best_loss = 1e10
    best_along_dice = 1e10
    best_along_train_loss = 1e10
    best_along_train_dice = 1e10
    loss_history = []

    ## for early stopping 
    early_stop = False
    patience = 10
    counter = 0

    for epoch in range(start_epoch,num_epochs+start_epoch):
        print('Starting epoch {}/{}'.format(epoch+1, num_epochs))

        # train
        model.train()

        metrics = defaultdict(float)
        epoch_size = 0
        
        # train model
        for batch_idx, (images, masks) in enumerate(train_loader):
            # print('img', images.size())
            images = images.to(device).float()
            masks = masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            # print('ch', np.unique(masks.cpu().numpy()))
            loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)

            save_metrics(metrics, images.size(0), loss, cross, dice)

            loss.backward()
            optimizer.step()

            # statistics
            epoch_size += images.size(0)

            if batch_idx % display_steps == 0:
                print('    ', end='')
                print('batch {:>3}/{:>3} cross: {:.4f} , dice {:.4f} , combined_loss {:.4f}\r'\
                      .format(batch_idx+1, len(train_loader), cross.item(), dice.item(),loss.item()))

            del images, masks, outputs, loss, cross, dice

        values = print_metrics(metrics, epoch_size, 'train')

        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))
        model.eval()

        # validate model
        for images, masks in val_loader:
            print('b', type(images))
            images = images.to(device).float()
            masks = masks.to(device).long()
            print('a', type(images))
            outputs = model(images)

            loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)

            save_metrics(metrics, images.size(0), loss, cross, dice)

            # statistics
            epoch_size += images.size(0)

            del images, masks, outputs, loss, cross, dice

        print_metrics(metrics, epoch_size, 'val')

        epoch_loss = metrics['loss'] / epoch_size
        epoch_dice = metrics['dice'] / epoch_size
        # save model if best validation loss
        if epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_along_dice = epoch_dice
            best_along_train_loss = values[0]
            best_along_train_dice = values[-1]
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.cpu()

            model_state_dict = model_copy.module.state_dict() if len(device_ids) > 1 else model_copy.state_dict()
            torch.save(model_state_dict, model_path)

            del model_copy

            counter = 0

        else:
            counter += 1
            print('EarlyStopping counter : {:>3} / {:>3}'.format(counter, patience))

            if counter >= patience :
                early_stop = True

        loss_history.append([best_loss,epoch_dice, best_along_train_loss, best_along_train_dice])
        print('Best val loss: {:4f} dice {:4f}, corresspond train loss {:4f} dice {:4f}'.format(
            best_loss, best_along_dice, best_along_train_loss, best_along_train_dice))

        if early_stop :
            print('Early Stopping')
            break
        
    return loss_history
    
    
if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    model = None
    
    if args.model == 'UNet':
        model = UNet(in_channels = n_channels, n_classes = n_classes)
    
    elif args.model == 'SegNet':
        model = SegNet(in_channels = n_channels, n_classes = n_classes)
        
    elif args.model == 'DenseNet':
        model = DenseNet(in_channels = n_channels, n_classes = n_classes)
    
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        raise SystemExit
        
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
        
    loss_history = train_net(model, args)
    
    # save validation loss history
    with open('./checkpoints/{}/validation_losses_{}'.format(args.data_path,args.model), 'w') as fp:
        # pickle.dump(loss_history, fp)
        fp.writelines('loss val {} dice {} train {} dice {}\n'.format(x[0],x[1],x[2],x[3]) for x in loss_history)
