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
import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
from optparse import OptionParser
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from util import make_one_hot
from dataset.dataset import AgDataset
from dataset.semantic_dataset import SampleDataset
from dataset.lung_dataset import CovidDataset
from dataset.scale_att_dataset import AttackDataset
from data import DefenseDataset, DefenseSclTestDataset
from model import UNet, SegNet, DenseNet
from model.AgNet.core.models import AG_Net
from loss import dice_score
from model import deeplab
# from model.Denoiser import get_net
from model.Denoiser_pvt import get_net
# from model.Denoiser_new import get_net
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim, ssim,  SSIM, MS_SSIM
# from piq import FSIM
import piq
from util import AverageMeter
import cv2
from model.AgNet.core.utils import get_model, dice_loss
from opts import get_args


def test(model, denoiser, args):
    
    data_path = args.data_path
    n_classes = args.classes
    suffix = args.suffix
    # args.output_path = '{}/{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.adv_model, args.attacks)
    # args.output_path = os.path.join(args.output_path, args.data_path,'512', args.model, args.adv_model,
    #                                 args.data_type, args.attacks)

    if args.attacks =='scl_attk':
        output_path = '{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.attacks)
        denoise_output = os.path.join(args.denoise_output, args.data_path, args.model,
                                        args.data_type, args.attacks, suffix)
    else:
        output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
                                        args.attacks,args.data_type, 'm'+ args.mask_type+'t'+args.target, suffix)
        denoise_output = os.path.join(args.denoise_output, args.data_path, args.model, args.adv_model,
                                        args.attacks, args.data_type, 'm'+ args.mask_type+'t'+args.target, suffix)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(denoise_output):
        os.makedirs(denoise_output)
    print('output path', args.output_path)
    print('denoised image path', args.denoise_output)
    device = torch.device(args.device)
        
    model = model.to(device)
    denoiser = denoiser.to(device)
    
    # set testdataset
        
    # test_dataset = SampleDataset(data_path,args.classes, args.channels, args.mode, None, args.adv_model, args.attacks,
    #                              args.target, args.data_type, args.width, args.height, args.mask_type, suffix)
    if args.attacks=='scl_attk':
        test_dataset = DefenseSclTestDataset(data_path, 'test', args.channels, data_type=args.data_type)
    elif any(attack in args.attacks for attack in ['dag','ifgsm']):
        if args.data_path=='brain':
            test_dataset = AgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                             args.attacks, args.data_type, args.mask_type,args.target, args.width, args.height,
            )
        elif args.data_path=='lung':
            test_dataset = CovidDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                         args.attacks, args.data_type, args.mask_type, args.target, args.width,
                                         args.height,
                                         )
        else:
            test_dataset = SampleDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                             args.attacks, args.data_type, args.mask_type,args.target, args.width, args.height,
            )
    else:
        test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)

    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))

    score = AverageMeter()

    ssims = []
    fsims = []
    adv_imgs = []
    defened_imgs = []

    model.eval()   # Set model to evaluate mode
    denoiser.eval()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.denoise_output):
        os.makedirs(args.denoise_output)
    # cm = plt.cm.jet
    cm = plt.get_cmap('gist_rainbow',1000)
    # cm= plt.get_cmap('viridis', 28)
    ssim_module = SSIM(data_range=1, size_average=False, channel=args.channels)
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            if args.attacks == 'scl_attk' or args.attacks=='semantic':
                target = make_one_hot(labels, n_classes, device)
            else:
                target = make_one_hot(labels[:, 0, :, :], n_classes, device)
            denoised_img = denoiser(inputs)
            denoised_img[denoised_img<0]=0
            denoised_img[denoised_img > 1] = 1
            # print('in', inputs.max(), denoised_img.min(), denoised_img.max())
            ssim_noise = ssim_module(inputs, denoised_img)
            # print('ssim', ssim_noise.shape)
            ssims.append(ssim_noise)
            # fsim_val = piq.fsim(inputs, denoised_img, data_range=1., reduction='none')
            # fsims.append(fsim_val)
            # print('max', torch.max(denoised_img))
            # for i, denoised_im in enumerate(denoised_img):
            #     # print(denoised_im.shape)
            #     cv2.imwrite(os.path.join(args.denoise_output,'{}.png'.format(batch_idx*args.batch_size+i)), denoised_im*255)
            # print('inpu', inputs.size())
            # denoised_img = svd_rgb(inputs,200,200,200)
            # denoised_img = svd_gs(inputs, 50)
            # AgNet
            if args.model == 'AgNet':
                denoised_img = denoised_img.double()
                out, side_5, side_6, side_7, side_8 = model(denoised_img)
                out = torch.log(softmax_2d(side_8) + EPS)
                # pred = model(inputs, train=False, defense=False)
                # print('pred size', denoised_img.size())
                out = torch.argmax(out, 1)
                # print('uot', out.size())
                masks = make_one_hot(out, n_classes, device)
                loss = 1-dice_loss(masks, target)
            else:
                pred = model(denoised_img)
                loss, masks = dice_score(pred,target)
            
            bsz = target.shape[0]
            score.update(loss.item(), bsz)
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
                output.save(os.path.join(output_path,'{}.png'.format(batch_idx*bsz+i)))
            denoised_img = np.moveaxis(denoised_img.cpu().numpy(),1, -1)
            for i, denoised_im in enumerate(denoised_img):
                # print(denoised_im.shape)
                cv2.imwrite(os.path.join(denoise_output,'{}.png'.format(batch_idx*bsz+i)), denoised_im*255)
            del inputs, labels, target,  loss

    avg_score = score.avg
    if args.model == 'AG_Net':
        avg_score = 1-avg_score
    
    print('dice_score : {:.4f}'.format(avg_score))
    avg_ssim = torch.mean(torch.cat(ssims))
    avg_ssim = avg_ssim.detach().cpu().numpy()
    avg_ssim = np.round(avg_ssim,2)
    print('avg ssim: ', avg_ssim)
    # avg_fsim = torch.mean(fsim_val).detach().cpu().numpy()
    # print('fsim', avg_fsim)

def onehot2norm(imgs):
    out = np.argmax(imgs,axis=1)
    return out

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    # model = None
    if args.model == 'UNet':
        model = UNet(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'SegNet':
        model = SegNet(in_channels = n_channels, n_classes = n_classes)

    elif args.model == 'DenseNet':
        model = DenseNet(in_channels = n_channels, n_classes = n_classes)
    elif args.model == 'AgNet':
        model = AG_Net(n_classes=n_classes,n_channels=args.channels, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
        model = model.double()
    else:
        model = deeplab.modeling.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride,
                                                      in_channels=args.channels, pretrained_backbone=False)
        if args.separable_conv and 'plus' in args.model:
            deeplab.convert_to_separable_conv(model.classifier)
        
    model_path = os.path.join(args.model_path, args.data_path, args.model + '.pth')
    # prefix = 'pgd'
    # prefix = 'ifgsm'
    # prefix = 'rd'
    # prefix = 'trf_rd'
    # guide_mode = 'SegNet'
    # prefix = 'pvt_scl'
    # prefix = 'pvt_scl_plus_leff'
    # prefix = 'pvt_semantic_plus_leff'
    prefix = args.suffix
    # prefix = 'pvt_scl_plus_leff_sub'
    guide_mode = 'UNet'
    # guide_mode = 'DenseNet'
    denoiser_path = os.path.join(args.denoiser_path, args.data_path, args.attacks, '{}_{}.pth'.format(guide_mode,prefix))
    # denoiser_path = os.path.join(args.denoiser_path, args.data_path, '{}.pth'.format(guide_mode))
    print('denoiser ', denoiser_path)
    # denoiser = get_net(args.height, args.width, args.classes, args.channels, denoiser_path, args.batch_size)
    denoiser = get_net(args.height, args.channels, denoiser_path)
    # model_path = os.path.join(args.model_path, 'fundus', args.model + '.pth')
    print('target model', model_path)
    model.load_state_dict(torch.load(model_path))
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

    test(model, denoiser, args)