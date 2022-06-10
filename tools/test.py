import numpy as np
import torch 
import torch.nn as nn
import os
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
import cv2
from optparse import OptionParser
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))

from model.AgNet.core.models import AG_Net
from util import make_one_hot
from dataset.dataset import SampleDataset, SegmentDataset
from dataset.lung_dataset import CovidDataset
from dataset.scale_att_dataset import AttackDataset
from model import UNet, SegNet, DenseNet
from loss import dice_score
from opts import get_args

def test(model, args):
    
    data_path = args.data_path
    gpu = args.gpus
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    suffix = args.suffix
    # args.output_path = '{}/{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.adv_model, args.attacks)
    # args.output_path = os.path.join(args.output_path, args.data_path,'512', args.model, args.adv_model,
    #                                 args.data_type, args.attacks)
    if args.mask_type !="":
        output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
                                    args.attacks, args.data_type,'m'+ args.mask_type+'t'+args.target, suffix)
    else:
        output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
                                        args.attacks, args.data_type, args.attacks, args.target)
    # output_path = os.path.join(args.output_path, args.data_path, args.model, args.mode, args.suffix)
    print('output path', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # adv_path = os.path.join(args.adv_path, args.data_path, args.model, args.mode, args.suffix)
    # if not os.path.exists(adv_path):
    #     os.makedirs(adv_path)
    # print('adv path', adv_path)
    device = torch.device(args.device)
        
    model = model.to(device)
    
    # set testdataset
    # test org
    # test_dataset = SampleDataset(data_path, n_classes, n_channels, mode= 'test',
    #         data_type='org',width=args.width,height=args.height)
    if args.attacks == 'scale_attk':
        test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)
        # test_dataset = AttackDataset(args.data_path, args.channels, 'train', args.data_path)
    elif 'lung' in data_path:
        test_dataset= CovidDataset(data_path, n_classes, n_channels, mode='test', model=args.model,
                        attack_type=args.attacks, data_type=args.data_type,mask_type=args.mask_type,\
                        target_class=args.target, width=args.width, height=args.height)
    else:
        # test adv
        test_dataset = SegmentDataset(data_path,args.classes, args.channels, args.mode, None, args.adv_model, args.attacks,
                    args.data_type,args.mask_type, args.target, args.width, args.height, suffix)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    avg_score = 0.0
    
    # test
    
    model.eval()   # Set model to evaluate mode

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # cm = plt.cm.jet
    cm = plt.get_cmap('gist_rainbow',1000)
    # cm= plt.get_cmap('viridis', 28)
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            bcz = inputs.shape[0]
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            # print('input', inputs.size())
            if args.attacks=='scale_attk':
                target = make_one_hot(labels[:, :, :], n_classes, device)
            else:
                target = make_one_hot(labels[:,0,:,:], n_classes, device)
            
            pred = model(inputs)
            # print('pred size', pred.size())
            loss, masks = dice_score(pred,target)
            
            avg_score += loss.data.cpu().numpy()
            # print('af',len(pred.data.cpu().numpy()))
            masks=onehot2norm(np.asarray(masks.data.cpu()))
            # inputs = torch.movedim(inputs,1,-1)
            # print('onehotcvt', masks.shape)
            for i,mask in enumerate(masks):
                # print(np.unique(mask))
                # mask=mask/255
                mask = mask / args.classes
                output=np.uint8(cm(mask)*255)
                # output = np.uint8(cm(mask) * args.classes)
                # print(np.unique(cm(mask)))
                # print('min,ma', np.min(output), np.max(output), output.shape)
                output= Image.fromarray(output)
                output.save(os.path.join(output_path,'{}.png'.format(batch_idx*bcz+i)))
                # adv = inputs[i]*255
                # adv = np.uint8(adv.data.cpu().numpy())
                # # adv = Image.fromarray(adv)
                # cv2.imwrite(os.path.join(adv_path,'{}.png'.format(batch_idx*args.batch_size+i)),adv)
            
            del inputs, labels, target, pred, loss
            
    avg_score /= len(test_loader)
    
    print('dice_score : {:.4f}'.format(avg_score))

def onehot2norm(imgs):
    out = np.argmax(imgs,axis=1)
    return out

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
    elif args.model == 'AgNet':
        model = AG_Net(n_classes=n_classes,n_channels=args.channels, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        # model = get_model('AG_Net')
        # model = model(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        model = model.double()
    else :
        print("wrong model : must be UNet, SegNet, or DenseNet")
        # raise SystemExit
        
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    print('model', args.model)
    model_path = os.path.join(args.model_path, args.data_path, args.model + '.pth')
    # model_path = os.path.join(args.model_path, 'fundus', args.model + '.pth')
    model.load_state_dict(torch.load(model_path))
    
    test(model, args)