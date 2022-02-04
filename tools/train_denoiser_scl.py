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
import time
from tqdm import trange, tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from os import path
from optparse import OptionParser
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from data import DefenseDataset, DefenseSclDataset
from model import UNet, SegNet, DenseNet
# from model.Denoiser import get_net
# from model.Denoiser_new import get_net
from model.Denoiser_pvt import get_net
from util import save_metrics, print_metrics
from loss import dice_loss1
from loss import onehot2norm, dice_loss, make_one_hot
# from pytorch_msssim import ms_ssim
# import pytorch_ssim


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
    parser.add_option('--gpus', dest='gpus',type='int',
                      default=1, help='gpu or cpu')
    parser.add_option('-n', '--nodes', default=1, type='int', metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_option('--lr', default=1e-3, dest='lr', type='float',
                        help='learning rate')
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
    parser.add_option('--device', dest='device', default=0, type='int',
                      help='device1 index number')

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
            
    device = torch.device(args.device)
    
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
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.1)
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

    # train_dataset = DefenseDataset(data_path, 'train', args.channels)

    train_dataset = DefenseSclDataset(data_path, 'train', args.channels)

    if 'fundus' in args.data_path:
        # val_dataset = DefenseDataset(data_path, 'val', args.channels)
        val_dataset = DefenseSclDataset(data_path, 'val', args.channels)
        train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
        val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
    else:
        train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.15, random_state=42)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
    print('total image : {}'.format(len(train_dataset) * 2))

    # train_size = len(train_dataset)*2
    # val_size = len(val_dataset)*2
    train_size = len(train_sampler)
    val_size = len(val_sampler)
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

    criterian = nn.MSELoss()
    # main train
    
    best_val_triplet_loss = 1e10
    best_triplet_train_loss = 1e10
    best_clean_dice = 1e10
    best_cln_dice = 1e10
    best_adv_dice = 1e10
    best_train_clean_dice = 1e10
    best_train_cln_dice = 1e10
    best_train_adv_dice = 1e10
    loss_history = []

    ## for early stopping 
    early_stop = False
    patience = 7
    counter = 0
    model.eval()
    for epoch in range(start_epoch,num_epochs+start_epoch):
        # model.eval()
        print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
        # train
        denoiser.train()

        metrics = defaultdict(float)
        clean_metrics = defaultdict(float)
        cln_metrics = defaultdict(float)
        denoised_metrics = defaultdict(float)
        epoch_size = 0
        # train model
        # total_loss = 0
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (images, adv_images, masks) in pbar:
            # print('img', images.size())
            images = images.to(device).float()
            adv_images = adv_images.to(device).float()
            masks = masks.to(device).long()
            # print('adv', adv_images.size(), images.max(), adv_images.max())
            # print('img', images.shape)
            # images = images.cuda(non_blocking=True).float()
            # adv_images = adv_images.cuda(non_blocking=True).float()
            # masks = masks.cuda(non_blocking=True).long()

            optimizer.zero_grad()

            # outputs, adv_outputs, l = model(device, images,adv_images)
            
            clean_outputs = model(images)
            clean_outputs_ = F.softmax(clean_outputs, dim=1)
            adv_outputs = model(adv_images)
            adv_outputs_ = F.softmax(adv_outputs, dim=1)

            # clean_outputs_ = clean_outputs_.to(device)
            denoised_advs = denoiser(adv_images)
            denoised_advs = denoised_advs.to(device)
            # print('denoise adv', denoised_advs.min(), denoised_advs.max())
            denoised_adv_output = model(denoised_advs)
            denoised_adv_output_ = F.softmax(denoised_adv_output, dim=1)
            # denoised_output_ = denoised_output_.cuda(gpu)
            # denoised_adv_output_ = denoised_adv_output_.to(device)

            denoised_cln = denoiser(images)
            denoised_cln = denoised_cln.to(device)
            denoised_cln_output = model(denoised_cln)
            denoised_cln_output_ = F.softmax(denoised_cln_output, dim=1)
            # denoised_cln_output_ = denoised_cln_output_.to(device)

            loss = criterian(clean_outputs_,denoised_cln_output_)+ criterian(clean_outputs_, denoised_adv_output_) #-\
            # criterian(clean_outputs_, adv_outputs_)
            # loss = criterian(clean_outputs_, denoised_adv_output_)
            loss.backward()
            optimizer.step()

            # print('mask', masks.squeeze(1).size(), outputs.size())
            dice = dice_loss1(clean_outputs, masks.squeeze(1), device, n_classes)
            cln_dice = dice_loss1(denoised_cln_output, masks.squeeze(1), device, n_classes)
            adv_dice = dice_loss1(denoised_adv_output, masks.squeeze(1), device, n_classes)
            # print('clean', dice.mean())
            # print('adv', adv_dice.mean())

            save_metrics(metrics, images.size(0), loss)
            save_metrics(clean_metrics, images.size(0), dice=dice)
            save_metrics(cln_metrics, images.size(0), dice=cln_dice)
            save_metrics(denoised_metrics, images.size(0), dice=adv_dice)
            epoch_size += images.size(0)
            ms = '[{:5d}/{:5d}]iter loss: {:.5f} clean dice {:.5f} cln dice {:.5f} adv dice {:.5f}'.format(
                batch_idx, len(train_loader), loss, dice, cln_dice, adv_dice
            )
            pbar.set_description(ms)
            # if batch_idx % display_steps == 0:
            #     print('    ', end='')
            #     print('batch {:>3}/{:>3} cross: {:.4f} , dice {:.4f} , dice_loss1 {:.4f}\r'\
            #           .format(batch_idx+1, len(train_loader), cross.item(), dice.item(),loss.item()))
            #
            # del images, masks, outputs, loss, cross, dice
            del images, masks,
        train_triplet_loss = metrics['loss'] / train_size
        train_clean_dice = clean_metrics['dice'] / train_size
        train_cln_dice = cln_metrics['dice'] / train_size
        train_adv_dice = denoised_metrics['dice'] / train_size
        print('train  triplet loss {:.8f} Clean dice: {:.4f} , Cln dice {:.4f} Adv dice {:.4f}'.
              format(train_triplet_loss, train_clean_dice, train_cln_dice, train_adv_dice))
        # values = print_metrics(metrics, epoch_size, 'train')
        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))
        denoiser.eval()
        metrics = defaultdict(float)
        clean_metrics = defaultdict(float)
        cln_metrics = defaultdict(float)
        denoised_metrics = defaultdict(float)
        # validate model
        pbar = tqdm(enumerate(val_loader))
        for batch_idx, (images, adv_images, masks) in pbar:
            images = images.to(device).float()
            adv_images = adv_images.to(device).float()
            masks = masks.to(device).long()
            # outputs, adv_outputs, l = model(device, images,adv_images)

            clean_outputs = model(images)
            clean_outputs_ = F.softmax(clean_outputs, dim=1)
            adv_outputs = model(adv_images)
            adv_outputs_ = F.softmax(adv_outputs, dim=1)

            # clean_outputs_ = clean_outputs_.to(device)
            denoised_advs = denoiser(adv_images)
            denoised_advs = denoised_advs.to(device)
            denoised_adv_output = model(denoised_advs)
            denoised_adv_output_ = F.softmax(denoised_adv_output, dim=1)
            # denoised_output_ = denoised_output_.cuda(gpu)
            # denoised_adv_output_ = denoised_adv_output_.to(device)

            denoised_cln = denoiser(images)
            denoised_cln = denoised_cln.to(device)
            denoised_cln_output = model(denoised_cln)
            denoised_cln_output_ = F.softmax(denoised_cln_output, dim=1)
            # denoised_output_ = denoised_output_.cuda(gpu)
            # denoised_cln_output_ = denoised_cln_output_.to(device)

            loss = criterian(clean_outputs_, denoised_cln_output_) + criterian(clean_outputs_, denoised_adv_output_) #- \
                   # criterian(clean_outputs_, adv_outputs_)

            dice = dice_loss1(clean_outputs, masks.squeeze(1), device, n_classes)
            cln_dice = dice_loss1(denoised_cln_output, masks.squeeze(1), device, n_classes)
            adv_dice = dice_loss1(denoised_adv_output, masks.squeeze(1), device, n_classes)

            save_metrics(metrics, images.size(0), loss)
            save_metrics(clean_metrics, images.size(0), dice=dice)
            save_metrics(cln_metrics, images.size(0), dice=cln_dice)
            save_metrics(denoised_metrics, images.size(0), dice= adv_dice)
            # save_metrics(metrics, images.size(0), loss, cross, dice)

            # statistics
            epoch_size += images.size(0)
            ms = '[{:5d}/{:5d}]iter loss: {:.5f} clean dice {:.5f} cln dice {:.5f} adv dice {:.5f}'.format(
                batch_idx, len(val_loader), loss, dice, cln_dice, adv_dice
            )
            pbar.set_description(ms)
            # del images, masks, outputs, loss, cross, dice
            del images, masks,
        val_triplet_loss = metrics['loss'] / val_size
        val_clean_dice = clean_metrics['dice'] / val_size
        val_cln_dice = cln_metrics['dice'] / val_size
        val_adv_dice = denoised_metrics['dice'] / val_size
        print('Val triplet loss {:.8f} Clean dice: {:.4f} , Cln dice {:.4f} Adv dice {:.4f}'.
              format(val_triplet_loss, val_clean_dice, val_cln_dice, val_adv_dice))
        # print_metrics(metrics, epoch_size, 'val')
        #
        # save model if best validation loss
        if val_triplet_loss < best_val_triplet_loss:
            print("saving best model")
            #total
            best_val_triplet_loss = val_triplet_loss
            best_triplet_train_loss = train_triplet_loss

            #dice
            best_clean_dice = val_clean_dice
            best_train_clean_dice = train_clean_dice
            
            best_cln_dice = val_cln_dice
            best_train_cln_dice = train_cln_dice
            
            best_adv_dice = val_adv_dice
            best_train_adv_dice = train_adv_dice

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
        print('Best val triplet : {:4f} corr train triplet  {:4f}|| dice val clean  {:4f} '
              'dice train clean  {:4f}|| cln val  {:4f} cln train  {:4f}|| adv val  {:4f} adv train  {:4f}||'
            .format(best_val_triplet_loss, best_triplet_train_loss, best_clean_dice, best_train_clean_dice, best_cln_dice,
                    best_train_cln_dice, best_adv_dice, best_train_adv_dice))

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
    denoiser= get_net(args.height, args.channels, args.resume)
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    # summary(denoiser, input_size=(n_channels, args.height, args.width), device='cpu')
    pytorch_total_params = sum(p.numel() for p in denoiser.parameters())
    print('Total params', pytorch_total_params)
    # os.environ['MASTER_ADDR'] = '10.2.142.212'
    # os.environ['MASTER_PORT'] = '8888'
    # args.world_size = args.gpus * args.nodes
    # mp.spawn(train_net, nprocs=args.gpus, args=(model, denoiser, args,))
    loss_history = train_net(model, denoiser, args)
