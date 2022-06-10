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
from os import path
from optparse import OptionParser
# sys.path.insert(0,path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))

from util import make_one_hot
from dataset.oct_dataset import SampleDataset
from dataset.scale_att_dataset import AttackDataset
from data import DefenseDataset, DefenseSclTestDataset
from dataset.semantic_dataset import DefenseSemanticTestDataset
from dataset.dataset import AgDataset
from dataset.lung_dataset import CovidDataset, CovidAgDataset
from model.AgNet.core.AgNet_Nonlocal import AG_NetNonlocal
from model.SegNet_Nonlocal import SegNetNonlocal
from model.UNet_Nonlocal import UNetNonlocal
from model.DenseNet_Nonlocal import DenseNetNonlocal
from loss import dice_score
from model import deeplab
import cv2
from model.AgNet.core.utils import dice_loss
from opts import get_args

def test(model, device, args):
    
    data_path = args.data_path
    n_classes = args.classes
    args.suffix = 'feature'
    suffix = args.suffix
    if 'scl' in args.attacks:
        output_path = os.path.join(args.output_path, args.data_path, args.model,
                                   args.attacks, args.data_type, suffix)
    else:
        output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
                                    args.attacks,args.data_type, 'm' + args.mask_type + 't' + args.target, suffix)
    print('output path', args.output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model.to(device)

    # test_dataset = SampleDataset(data_path,args.classes, args.channels, args.mode, None, args.adv_model, args.attacks,
    #                              args.target, args.data_type, args.width, args.height, args.mask_type, suffix)
    if args.attacks == 'scl_attk':
        test_dataset = DefenseSclTestDataset(data_path, 'test', args.channels, data_type=args.data_type)
    elif any(attack in args.attacks for attack in ['dag', 'ifgsm']):
        if 'lung' in data_path:
            if args.model=='AgNet':
                test_dataset = CovidAgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                            args.attacks, args.data_type, args.mask_type, args.target, args.width,
                                            args.height)
            else:
                test_dataset= CovidDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                 args.attacks, args.data_type, args.mask_type,  args.target, args.width, args.height)
        else:
            test_dataset = SampleDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                 args.attacks, args.data_type, args.mask_type,  args.target, args.width, args.height)
    else:
        print('load semantic dataset')
        # test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)
        if args.data_path=='brain':
            test_dataset = AgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                             args.attacks, args.target, args.attacks, args.width, args.height, args.mask_type,
                             suffix=args.suffix)
        else:
            test_dataset = DefenseSemanticTestDataset(data_path, args.mode, args.channels, args.model, \
                                     args.attacks, args.target, args.mask_type)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    avg_score = 0.0
    
    model.eval()   # Set model to evaluate mode
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.denoise_output):
        os.makedirs(args.denoise_output)
    # cm = plt.cm.jet
    cm = plt.get_cmap('gist_rainbow',1000)
    # cm= plt.get_cmap('viridis', 28)
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            bcz = inputs.shape[0]
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            if args.model=='AgNet' or args.attacks == 'scl_attk':
                target = make_one_hot(labels, n_classes, device)
            else:
                target = make_one_hot(labels[:, 0, :, :], n_classes, device)
            # print('max', torch.max(denoised_img))
            # for i, denoised_im in enumerate(denoised_img):
            #     # print(denoised_im.shape)
            #     cv2.imwrite(os.path.join(args.denoise_output,'{}.png'.format(batch_idx*args.batch_size+i)), denoised_im*255)
            # print('inpu', inputs.size())
            # denoised_img = svd_rgb(inputs,200,200,200)
            # denoised_img = svd_gs(inputs, 50)
            # AgNet
            if args.model == 'AgNet':
                inputs = inputs.double()
                out, side_5, side_6, side_7, side_8 = model(inputs)
                out = torch.log(softmax_2d(side_8) + EPS)
                # pred = model(inputs, train=False, defense=False)
                # print('pred size', denoised_img.size())
                out = torch.argmax(out, 1)
                # print('uot', out.size())
                masks = make_one_hot(out, n_classes, device)
                loss = 1-dice_loss(masks, target)
            else:
                inputs = inputs.float()
                pred = model(inputs)
                loss, masks = dice_score(pred,target)
            
            avg_score += loss.data.cpu().numpy()
            # print('af',len(pred.data.cpu().numpy()))
            masks=onehot2norm(np.asarray(masks.data.cpu()))
            # AG
            # masks = out.data.cpu()
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
            del inputs, labels, target,  loss
            
    avg_score /= len(test_loader)
    if args.model == 'AgNet':
        print('invert')
        avg_score = 1-avg_score
    
    print('dice_score : {:.4f}'.format(avg_score))

def onehot2norm(imgs):
    out = np.argmax(imgs,axis=1)
    return out

if __name__ == "__main__":

    args = get_args()
    device = torch.device(args.device)
    n_channels = args.channels
    n_classes = args.classes
    
    if args.model == 'UNet':
        model = UNetNonlocal(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'SegNet':
        model = SegNetNonlocal(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'DenseNet':
        model = DenseNetNonlocal(in_channels = n_channels, n_classes = n_classes)
    elif args.model == 'AgNet':
        model = AG_NetNonlocal(n_classes=n_classes,n_channels=args.channels, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        model = model.double()
    else:
        model = deeplab.modeling_nonlocal.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride,
                                                      in_channels=args.channels, pretrained_backbone=False)
        if args.separable_conv and 'plus' in args.model:
            deeplab.convert_to_separable_conv(model.classifier)
    # else :
    #     print("wrong model : must be UNet, SegNet, or DenseNet")
    #     raise SystemExit
        
    suffix = 'feature'
    # prefix = args.suffix
    guide_mode = 'UNet'
    # guide_mode = 'DenseNet'
    denoiser_path = os.path.join(args.denoiser_path, args.data_path, args.attacks, args.model+'_{}.pth'.format(suffix))
    # denoiser_path = os.path.join(args.denoiser_path, args.data_path, 'UNet.pth')
    print('denoiser ', denoiser_path)
    model.load_state_dict(torch.load(denoiser_path, map_location=device))
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

    test(model, device, args)