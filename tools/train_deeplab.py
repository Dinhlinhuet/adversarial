from tqdm import tqdm
import os
import random
import argparse
import numpy as np
import time
import copy
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Sampler
from matplotlib import pyplot as plt
from os import path
import sys
from PIL import Image
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from model import deeplab
import utils
from dataset.dataset import SampleDataset, SegmentDataset
from dataset.lung_dataset import CovidDataset
from util import save_metrics, print_metrics
from util import save_metrics, print_metrics, make_one_hot
from loss import combined_loss, dice_score


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument('--data_path', dest='data_path',type=str,
                      default='data/samples', help='data path')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                      help='number of epochs')
    parser.add_argument("--classes", type=int, default=1,
                        help="num classes (default: None)")
    parser.add_argument('--channels', dest='channels', default=3, type=int,
                      help='number of channels')
    parser.add_argument('--width', dest='width', default=256, type=int,
                      help='image width')
    parser.add_argument('--height', dest='height', default=256, type=int,
                      help='image height')

    # Deeplab Options
    # available_models = sorted(name for name in deeplab.modeling.__dict__ if name.islower() and \
    #                           not (name.startswith("__") or name.startswith('_')) and callable(
    #                           deeplab.modeling.__dict__[name])
    #                           )
    # parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
    #                     choices=available_models, help='model name')
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    

    # Train Options
    parser.add_argument('--output_path', dest='output_path', type=str,
                      default='./output', help='output_path')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                      help='manual epoch number (useful on restarts)')
    parser.add_argument('--save-dir', default='./checkpoints/', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument('--device', dest='device', default=0, type=int,
                      help='device1 index number')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    return parser

def validate(model, loader, device, n_classes, metrics):
    """Do validation and return specified samples"""
    with torch.no_grad():
        epoch_size = 0
        for i, (images, masks) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            # print('output', outputs.shape, outputs.max(), outputs.min())
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            # print('pred', preds.shape, preds.max(), preds.min())
            loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)
            # print('loss', loss, cross, dice)
            save_metrics(metrics, images.size(0), loss, cross, dice)
            epoch_size+=images.size(0)
    # print_metrics(metrics, epoch_size, 'val')
    return metrics, epoch_size

def onehot2norm(imgs):
    out = np.argmax(imgs,axis=1)
    return out

def main():
    args = get_argparser().parse_args()
    device = torch.device(args.device)
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    data_path = args.data_path
    num_epochs = args.epochs
    n_classes = args.classes
    n_channels = args.channels
    #for fundus and brain
    if not 'fundus' in data_path:
        print('datapath', data_path)
        train_dataset = SampleDataset(data_path, n_classes, n_channels, mode='train',
            data_type='org',width=args.width,height=args.height)
        train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
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
    else:
        if 'lung' in data_path:
            train_dataset = CovidDataset(data_path, n_classes, n_channels, mode='train', model=None,
                                         data_type='org', width=args.width, height=args.height)
            val_dataset = CovidDataset(data_path, n_classes, n_channels, mode='val', model=None,
                                       data_type='org', width=args.width, height=args.height)
        else:
            train_dataset = SegmentDataset(data_path, n_classes, n_channels, mode= 'train', gen_mode=None,model=None,
                type=None,target_class=None,data_type='org',width=args.width,height=args.height, mask_type=None, suffix=None)
            val_dataset = SegmentDataset(data_path, n_classes, n_channels, mode= 'val', gen_mode=None,model=None,
                type=None,target_class=None,data_type='org',width=args.width,height=args.height, mask_type=None, suffix=None)
        train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
        val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
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
    print('total train image : {}'.format(len(train_sampler)))
    print('total val image : {}'.format(len(val_sampler)))
    # test_dataset = SegmentDataset(data_path, n_classes, n_channels, mode= 'test', gen_mode=None,model=None,
    #     type=None,target_class=None,data_type='org',width=args.width,height=args.height, mask_type=None, suffix=None)
    # test_sampler = SubsetRandomSampler(np.arange(len(test_dataset)))
    # print('total test image : {}'.format(len(test_sampler)))


    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     sampler=test_sampler
    # )
    print('num class', args.classes, 'out stride', args.output_stride)
    # Set up model (all models are 'constructed at deeplab.modeling)
    model = deeplab.modeling.__dict__[args.model](num_classes=args.classes, in_channels=args.channels,
                                                  output_stride=args.output_stride, pretrained_backbone=False)
    # model.backbone.in_channels = 1
    # model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    if args.separable_conv and 'plus' in args.model:
        deeplab.convert_to_separable_conv(model.classifier)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
    if args.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    utils.mkdir('checkpoints')
    # Restore
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        if args.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("Training state restored from %s" % args.ckpt)
        print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model.to(device)

    # criterion = utils.get_loss(opts.loss_type)
    if args.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

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
    model_folder = os.path.abspath('./{}/{}/'.format(save_dir, args.data_path))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, args.model+'.pth')
    print('save path', model_path)

    best_loss = 1e10
    best_along_dice = 1e10
    best_along_train_loss = 1e10
    best_along_train_dice = 1e10
    loss_history = []

    ## for early stopping
    early_stop = False
    patience = 10
    counter = 0

    for epoch in range(start_epoch, num_epochs + start_epoch):
        # =====  Train  =====
        model.train()

        metrics = defaultdict(float)
        epoch_size = 0
        for (images, masks) in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)
            # print('input', images.size(), images.max())
            optimizer.zero_grad()
            outputs = model(images)
            loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)
            save_metrics(metrics, images.size(0), loss, cross, dice)
            # dice.backward()
            # cross.backward()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_size += images.size(0)

        values = print_metrics(metrics, epoch_size, 'train')

        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch + 1))
        metrics = defaultdict(float)
        model.eval()
        metrics, epoch_size = validate(
            model=model, loader=val_loader, device=device, n_classes=n_classes, metrics=metrics)
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

            model_state_dict = model_copy.state_dict()
            torch.save(model_state_dict, model_path)

            del model_copy

            counter = 0

        else:
            counter += 1
            print('EarlyStopping counter : {:>3} / {:>3}'.format(counter, patience))

            if counter >= patience:
                early_stop = True

        loss_history.append([best_loss, epoch_dice, best_along_train_loss, best_along_train_dice])
        print('Best val loss: {:4f} dice {:4f}, corresspond train loss {:4f} dice {:4f}'.format(
            best_loss, best_along_dice, best_along_train_loss, best_along_train_dice))

        if early_stop:
            print('Early Stopping')
            break
    # test(model, test_loader, device, n_classes, args)

def test(model, test_loader, device, n_classes,args):

    epoch_size = 0
    metrics = defaultdict(float)
    model.eval()
    output_path = os.path.join(args.output_path, args.data_path, 'DeepLabplus')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output path', output_path)
    # cm = plt.cm.jet
    cm = plt.get_cmap('gist_rainbow', 1000)
    # cm= plt.get_cmap('viridis', 28)
    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(test_loader):

            images = images.to(device).float()
            labels = labels.to(device).long()
            # masks = labels
            # print('input', images.size(), images.max())
            # if args.suffix == 'scl_attk':
            #     target = make_one_hot(labels[:, :, :], n_classes, device)
            # else:
            #     target = make_one_hot(labels[:, 0, :, :], n_classes, device)
            target = make_one_hot(labels.squeeze(1), n_classes, device)
            pred = model(images)
            # print('pred size', pred.size(), pred.max(), pred.min())
            dice, masks = dice_score(pred, target)
            # loss, cross, dice = combined_loss(pred, labels.squeeze(1), device, n_classes)
            # save_metrics(metrics, images.size(0), loss, cross, dice)

            save_metrics(metrics, images.size(0), dice)
            epoch_size += images.size(0)
            # print('af',len(pred.data.cpu().numpy()))
            masks = onehot2norm(np.asarray(masks.data.cpu()))
            # print('onehotcvt', masks.shape)
            for i, mask in enumerate(masks):
                # print(np.unique(mask))
                # mask=mask/255
                mask = mask / args.classes
                output = np.uint8(cm(mask) * 255)
                # output = np.uint8(cm(mask) * args.classes)
                # print(np.unique(cm(mask)))
                # print('min,ma', np.min(output), np.max(output), output.shape)
                output = Image.fromarray(output)
                output.save(os.path.join(output_path, '{}.png'.format(batch_idx * args.batch_size + i)))

            del images, labels,  pred, dice

    # avg_score = metrics['loss'] / epoch_size
    avg_dice = metrics['dice'] / epoch_size
    # print('loss : {:.4f}'.format(1-avg_score))
    print('dice_score : {:.4f}'.format(avg_dice))
    print_metrics(metrics, epoch_size, 'test')

if __name__ == '__main__':
    main()
