from tqdm import tqdm
import os
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
import random
import argparse
import numpy as np
import time
import copy
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader,Sampler

from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image

from model import deeplab
from dataset.dataset import SampleDataset, SegmentDataset
from dataset.scale_att_dataset import AttackDataset
from util import save_metrics, print_metrics, make_one_hot
from loss import combined_loss, dice_score
import utils


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument('--data_path', dest='data_path',type=str,
                      default='data/samples', help='data path')
    parser.add_argument('--epochs', dest='epochs', default=50, type=int,
                      help='number of epochs')
    parser.add_argument("--classes", type=int, default=1,
                        help="num classes (default: None)")
    parser.add_argument('--channels', dest='channels', default=3, type=int,
                      help='number of channels')
    parser.add_argument('--width', dest='width', default=256, type=int,
                      help='image width')
    parser.add_argument('--height', dest='height', default=256, type=int,
                      help='image height')

    available_models = sorted(name for name in deeplab.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              deeplab.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    

    # test Options
    parser.add_argument('--output_path', dest='output_path', type=str,
                      default='./output', help='output_path')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument('--suffix', dest='suffix', type=str,
                      default='', help='suffix to purpose')
    parser.add_argument("--ckpt", default='./checkpoints/', type=str,
                        help="restore from checkpoint")
    parser.add_argument('--device', dest='device', default=0, type=int,
                      help='device1 index number')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    return parser

def validate(model, loader, device, n_classes, metrics):
    """Do validation and return specified samples"""
    with torch.no_grad():
        for i, (images, masks) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            # print('output', outputs.shape, outputs.max(), outputs.min())
            # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            # print('pred', preds.shape, preds.max(), preds.min())
            loss, cross, dice = combined_loss(outputs, masks.squeeze(1), device, n_classes)
            save_metrics(metrics, images.size(0), loss, cross, dice)



def test():
    args = get_argparser().parse_args()
    device = torch.device(args.device)
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    data_path = args.data_path
    n_classes = args.classes
    n_channels = args.channels
    #for fundus and brain
    if 'octa' in data_path:
        # test_dataset = SampleDataset(data_path, n_classes, n_channels, mode= 'test',
        #     data_type='org',width=args.width,height=args.height)
        test_dataset = AttackDataset(args.data_path, args.channels, 'test', args.data_path)
        test_sampler = SubsetRandomSampler(np.arange(len(test_dataset)))
    else:
        # test_dataset = SegmentDataset(data_path, n_classes, n_channels, mode= 'test', gen_mode=None,model=None,
        #     type=None,target_class=None,data_type='org',width=args.width,height=args.height, mask_type=None, suffix=None)
        test_dataset = AttackDataset(args.data_path, args.channels, 'test', args.data_path)
        test_sampler = SubsetRandomSampler(np.arange(len(test_dataset)))
    print('total test image : {}'.format(len(test_sampler)))


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=test_sampler
    )

    print('num class', args.classes, 'out stride', args.output_stride)
    # Set up model (all models are 'constructed at deeplab.modeling)
    model = deeplab.modeling.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride,
                                                  in_channels=args.channels, pretrained_backbone=False)
    if args.separable_conv and 'plus' in args.model:
        deeplab.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)

    # model = nn.DataParallel(model)
    model.to(device)
    model_path = os.path.join(args.ckpt, args.data_path, args.model+'.pth')
    print('load model ', model_path)
    checkpoint = torch.load(model_path)
    # print('checkpoint', checkpoint.keys())
    model.load_state_dict(checkpoint)

    epoch_size = 0
    metrics = defaultdict(float)
    model.eval()
    output_path = os.path.join(args.output_path, args.data_path, args.model)
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

            save_metrics(metrics, images.size(0), dice=dice)
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

def onehot2norm(imgs):
    out = np.argmax(imgs,axis=1)
    return out

if __name__ == '__main__':
    test()
