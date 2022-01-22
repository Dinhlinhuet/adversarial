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
from loss import onehot2norm, dice_loss, make_one_hot
import copy
import pickle
import time
import torch.multiprocessing as mp
import torch.distributed as dist

from optparse import OptionParser

from data import DefenseDataset
from model import UNet, SegNet, DenseNet
# from model.Denoiser import get_net
from model.Denoiser_new import get_net
from util import save_metrics, print_metrics
from loss import combined_loss
from pytorch_msssim import ms_ssim
# import pytorch_ssim


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--epochs', dest='epochs', default=50, type='int',
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
    parser.add_option('--gpus', dest='gpus',type='int',
                      default=1, help='gpu or cpu')
    parser.add_option('-n', '--nodes', default=1, type='int', metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_option('--nr', default=0, dest='nr', type='int',
                        help='ranking within the nodes')
    parser.add_option('--suffix', dest='suffix', type='string',
                      default='', help='suffix to purpose')
    parser.add_option('--resume', default='', type='string', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_option('--target_model', default='./checkpoints/', type='string', metavar='PATH',
                      help='path to target model (default: none)')
    parser.add_option('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_option('--save-dir', default='./checkpoints/denoiser/', type='string', metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
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

def train_net(model, denoiser, args):
    # rank = args.nr * args.gpus + gpu
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    data_path = args.data_path
    num_epochs = args.epochs
    n_classes = args.classes
    # data_width = args.width
    # data_height = args.height

    # set device configuration
    device_ids = []

    if args.gpus >= 1 :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        # device = torch.device(args.device1)
        device = torch.device('cuda')
        device_ids.append(args.device1)
        if args.device2 != -1 :
            print('device2', args.device2)
            device_ids.append(args.device2)
            device2 = torch.device('cuda:1')
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)

    else :
        device = torch.device("cpu")
    
    # if len(device_ids) > 1:
    #     print('parallel')
    #     # model = nn.DataParallel(model, device_ids = device_ids)
    #     denoiser = nn.DataParallel(denoiser, device_ids=device_ids)

    if isinstance(denoiser, DataParallel):
        print('parallely')
        params = denoiser.module.parameters()
    else:
        params = denoiser.parameters()
    # torch.cuda.set_device(gpu)
    # denoiser.cuda(gpu)
    # denoiser = nn.parallel.DistributedDataParallel(denoiser, device_ids=[gpu])
    # pytorch_total_params = sum(p.numel() for p in c_params)
    # print('total denoiser params', pytorch_total_params)
    optimizer = torch.optim.Adam(params)
    model = model.to(device)
    # model.cuda(gpu)
    denoiser = denoiser.to(device)
    for param in model.parameters():
        param.requires_grad = False
    # ssim_module = pytorch_ssim.SSIM(window_size=11)
    # ssim_module = SSIM(data_range=1, size_average=True, channel=3)
    # ssim_module = ssim_module.to(device)
    # loss = loss.to(device)

    # set image into training and validation dataset

    train_dataset = DefenseDataset(data_path, 'train', args.channels)

    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    if 'fundus' in args.data_path:
        val_dataset = DefenseDataset(data_path, 'val', args.channels)
        train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
        val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
    print('total image : {}'.format(len(train_dataset) * 2))

    # train_sampler = DistributedSampler(train_dataset,num_replicas=args.world_size, rank=rank)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)
    # train_size = len(train_dataset)*2
    # val_size = len(val_dataset)*2
    train_size = len(train_sampler) * 2
    val_size = len(val_sampler)*2
    print('train',train_size)
    print('val',val_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=train_sampler
    )

    if 'fundus' in args.data_path:
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
    save_dir = args.save_dir
    if args.resume:
        checkpoint = torch.load(args.resume)
        save_dir = args.save_dir
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
    denoiser_folder = os.path.abspath('./{}/{}/'.format(save_dir,args.data_path))
    if not os.path.exists(denoiser_folder):
        os.makedirs(denoiser_folder)
    
    # if args.model == 'UNet':
    #     denoiser_path = os.path.join(denoiser_folder, 'UNet.pth')
    #
    # elif args.model == 'SegNet':
    #     denoiser_path = os.path.join(denoiser_folder, 'SegNet.pth')
    #
    # elif args.model == 'DenseNet':
    denoiser_path = os.path.join(denoiser_folder,'{}_{}.pth'.format(args.model, args.suffix))
    # denoiser_path = os.path.join(denoiser_folder, args.model + '_pgd.pth')
    print('save path', denoiser_path)
    # set optimizer


    # main train
    
    best_total_loss = 1e10
    best_along_total_train_loss = 1e10
    best_l1_loss = 1e10
    best_along_dice = 1e10
    best_along_train_l1_loss = 1e10
    best_along_train_dice = 1e10
    best_along_cb = 1e10
    best_along_train_cb = 1e10
    loss_history = []

    ## for early stopping 
    early_stop = False
    patience = 7
    counter = 0
    model.eval()
    # if isinstance(denoiser, DataParallel):
    #     denoiser.module.train()
    # else:
    #     denoiser.train()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch,num_epochs+start_epoch):
        # model.eval()
        print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
        # train
        # model.train()
        denoiser.train()

        metrics = defaultdict(float)
        denoised_metrics = defaultdict(float)
        epoch_size = 0
        l1_loss = []
        # train model
        # total_loss = 0
        # total_loss = []
        for batch_idx, (images, adv_images, masks) in enumerate(train_loader):
            # print('img', images.size())
            images = images.to(device).float()
            adv_images = adv_images.to(device).float()
            masks = masks.to(device).long()

            # images = images.cuda(non_blocking=True).float()
            # adv_images = adv_images.cuda(non_blocking=True).float()
            # masks = masks.cuda(non_blocking=True).long()

            optimizer.zero_grad()

            # outputs, adv_outputs, l = model(device, images,adv_images)
            
            clean_outputs = model(images)
            clean_outputs_ = F.softmax(clean_outputs, dim=1)
            # clean_outputs_ = clean_outputs_.cuda(gpu)
            # clean_outputs = clean_outputs.to(device)
            clean_outputs_ = clean_outputs_.to(device)
            denoised_images = denoiser(adv_images)
            denoised_images = denoised_images.to(device)
            denoised_output = model(denoised_images)
            denoised_output_ = F.softmax(denoised_output, dim=1)
            # denoised_output_ = denoised_output_.cuda(gpu)
            denoised_output_ = denoised_output_.to(device)
            l = denoiser.loss(clean_outputs_,denoised_output_)
            # ssim_loss = ssim_module(images,denoised_images)
            # ssim_loss = ms_ssim(images, denoised_images, data_range=1, size_average=False)
            # ssim_loss = torch.mean(ssim_loss)
            # ssim_loss = ssim(images,denoised_images, data_range=1, size_average=True)
            # outputs, adv_outputs = model(images, adv_images)
            # total_loss += ssim_loss
            # total_loss.append(ssim_loss)
            total_loss = 0
            for ll in l:
                # print('ll', ll.mean().data.cpu().numpy())
                l_mean = ll.mean()
                total_loss += l_mean
                ll = l_mean.data.cpu().numpy()
                l1_loss.append(ll)
            # print('mask', masks.squeeze(1).size(), outputs.size())
            cb_loss, cross, dice = combined_loss(clean_outputs, masks.squeeze(1), device, n_classes)
            adv_cb_loss, adv_cross, adv_dice = combined_loss(denoised_output, masks.squeeze(1), device, n_classes)
            # print('clean', dice.mean())
            # print('adv', adv_dice.mean())
            # adv_outputs_ = F.softmax(adv_outputs, dim=1)
            # adv_outputs_ = onehot2norm(adv_outputs_, device)
            # l = dice_loss(outputs, adv_outputs_)
            # adv_outputs_ = make_one_hot(masks.squeeze(1), n_classes, device)
            # print('adv', adv_outputs_.type())
            # l = dice_loss(outputs, adv_outputs_)
            # l1_loss.append(l)
                # loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)

            save_metrics(metrics, images.size(0), cb_loss, cross, dice)
            save_metrics(denoised_metrics, images.size(0), adv_cb_loss, adv_cross, adv_dice)
            # final_loss = adv_cb_loss+total_loss
            # loss.backward()
            # cb_loss.backward(retain_graph=True)
            # adv_cb_loss.backward(retain_graph=True)
            # l.backward()
            # ssim_loss.backward()
            # adv_cb_loss.backward()
            total_loss.backward()
            # final_loss.backward()
            optimizer.step()

            # statistics
            epoch_size += images.size(0)

            # if batch_idx % display_steps == 0:
            #     print('    ', end='')
            #     print('batch {:>3}/{:>3} cross: {:.4f} , dice {:.4f} , combined_loss {:.4f}\r'\
            #           .format(batch_idx+1, len(train_loader), cross.item(), dice.item(),loss.item()))
            #
            # del images, masks, outputs, loss, cross, dice
            del images, masks, denoised_images
        train_l1_loss = sum(l1_loss)/len(l1_loss)
        # train_ssim_loss = sum(total_loss)/len(train_loader)
        # train_cb_loss = metrics['loss'] / train_size
        # train_dice = metrics['dice'] / train_size
        train_dn_cb_loss = denoised_metrics['loss'] / train_size
        train_dn_dice = denoised_metrics['dice'] / train_size
        print('train  L1 loss {:.8f} combine loss: {:.4f} , dice {:.4f}'.format(train_l1_loss, train_dn_cb_loss,
                                                                                train_dn_dice))
        # values = print_metrics(metrics, epoch_size, 'train')
        l1_loss = []
        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))
        denoiser.eval()
        metrics = defaultdict(float)
        denoised_metrics = defaultdict(float)
        # val_total_loss = 0
        # val_total_loss = []
        # validate model
        for images, adv_images, masks in val_loader:
            images = images.to(device).float()
            adv_images = adv_images.to(device).float()
            masks = masks.to(device).long()
            # print('adv', adv_images.size())
            # outputs, adv_outputs, l = model(device, images,adv_images)
            
            clean_outputs = model(images)
            clean_outputs_ = F.softmax(clean_outputs, dim=1)
            # clean_outputs = clean_outputs.to(device)
            clean_outputs_ = clean_outputs_.to(device)
            # clean_outputs_ = clean_outputs_.cuda(gpu)

            denoised_images = denoiser(adv_images)
            denoised_images = denoised_images.to(device)

            denoised_output = model(denoised_images)
            denoised_output_ = F.softmax(denoised_output, dim=1)
            # denoised_output_ = denoised_output_.cuda(gpu)
            denoised_output_ = denoised_output_.to(device)
            l = denoiser.loss(clean_outputs_, denoised_output_)

            # ssim_loss = ssim_module(images, denoised_images)
            # ssim_loss = ms_ssim(images, denoised_images, data_range=1, size_average=False)
            # ssim_loss = torch.mean(ssim_loss)
            # # ssim_loss = ssim(images, denoised_images, data_range=1, size_average=True)
            # # val_total_loss += ssim_loss
            # val_total_loss.append(ssim_loss)

            total_loss = 0
            for ll in l:
                # print('ll', ll.mean().data.cpu().numpy())
                l_mean = ll.mean()
                total_loss += l_mean
                ll = l_mean.data.cpu().numpy()
                l1_loss.append(ll)
            cb_loss, cross, dice = combined_loss(clean_outputs, masks.squeeze(1), device, n_classes)
            adv_cb_loss, adv_cross, adv_dice = combined_loss(denoised_output, masks.squeeze(1), device, n_classes)

            save_metrics(metrics, images.size(0), cb_loss, cross, dice)
            save_metrics(denoised_metrics, images.size(0), adv_cb_loss, adv_cross, adv_dice)
            # save_metrics(metrics, images.size(0), loss, cross, dice)

            # statistics
            epoch_size += images.size(0)

            # del images, masks, outputs, loss, cross, dice
            del images, masks, denoised_images
        val_l1_loss = sum(l1_loss) / len(l1_loss)
        # val_ssim_loss = val_total_loss/len(val_loader)
        # val_ssim_loss = sum(val_total_loss) / len(val_loader)
        # val_cb_loss = metrics['loss'] / val_size
        # val_dice = metrics['dice'] / val_size
        val_dn_cb_loss = denoised_metrics['loss'] / val_size
        val_dn_dice = denoised_metrics['dice'] / val_size
        # val_total_loss = val_l1_loss+val_cb_loss
        # val_total_loss = val_ssim_loss+ val_dn_cb_loss
        val_total_loss = val_l1_loss
        print('val L1 loss {:.8f} combine loss: {:.4f} , dice {:.4f} total loss {:.4f}'.format(val_l1_loss,
                                                                        val_dn_cb_loss, val_dn_dice, val_total_loss))
        # print_metrics(metrics, epoch_size, 'val')
        #
        # save model if best validation loss
        if val_total_loss < best_total_loss:
            print("saving best model")
            #total
            best_total_loss = val_total_loss
            best_along_total_train_loss = train_l1_loss+train_dn_cb_loss

            #ssim
            best_ssim_loss = val_l1_loss
            best_along_train_ssim_loss = train_l1_loss
            #l1
            best_l1_loss = val_l1_loss
            best_along_train_l1_loss = train_l1_loss
            #dice
            best_along_dice = val_dn_dice
            best_along_train_dice = train_dn_dice
            #cb
            best_along_cb = val_dn_cb_loss
            best_along_train_cb = train_dn_cb_loss

            model_copy = copy.deepcopy(denoiser)
            model_copy = model_copy.cpu()

            model_state_dict = model_copy.module.state_dict() if isinstance(denoiser, DataParallel) else model_copy.state_dict()
            torch.save(model_state_dict, denoiser_path)

            del model_copy

            counter = 0

        else:
            counter += 1
            print('EarlyStopping counter : {:>3} / {:>3}'.format(counter, patience))

            if counter >= patience :
                early_stop = True

        loss_history.append([ best_total_loss, best_l1_loss, best_along_dice, best_along_total_train_loss,
                              best_along_train_l1_loss, best_along_train_dice
                             ])
        print('Best val L1 loss: {:4f} corr train L1 loss {:4f}|| dice val loss {:4f} dice train loss {:4f}||'
              'combine val loss {:4f} combine train loss {:4f}'
            .format(best_l1_loss, best_along_train_l1_loss, best_along_dice, best_along_train_dice,
                    best_along_cb, best_along_train_cb))

        if early_stop :
            print('Early Stopping')
            break
        
    return loss_history
    
    
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
        model = UNet(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'SegNet':
        model = SegNet(in_channels=n_channels, n_classes=n_classes)

    elif args.model == 'DenseNet':
        model = DenseNet(in_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(torch.load(args.target_model))
    denoiser_path = os.path.join(args.save_dir, args.data_path, args.model+'.pth')
    denoiser= get_net(args.height, args.width, n_classes, args.channels, args.resume, args.batch_size)
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    summary(denoiser, input_size=(n_channels, args.height, args.width), device='cpu')
    # os.environ['MASTER_ADDR'] = '10.2.142.212'
    # os.environ['MASTER_PORT'] = '8888'
    # args.world_size = args.gpus * args.nodes
    # mp.spawn(train_net, nprocs=args.gpus, args=(model, denoiser, args,))
    loss_history = train_net(model, denoiser, args)
    
    # # save validation loss history
    # with open('./checkpoints/denoiser/{}/validation_losses_{}'.format(args.data_path,args.model), 'w') as fp:
    #     # pickle.dump(loss_history, fp)
    #     fp.writelines('loss val: total {} l1 {} dice {} ; train: total {} l1 {} dice {}\n'.format(x[0],x[1],x[2],x[3],x[4],x[5]) for x in loss_history)
