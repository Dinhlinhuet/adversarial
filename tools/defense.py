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
# from dataset.dataset import SampleDataset
from dataset.scale_att_dataset import AttackDataset
from data import DefenseDataset, DefenseSclTestDataset
from model import UNet, SegNet, DenseNet
from loss import dice_score
from model import deeplab
# from model.Denoiser import get_net
from model.Denoiser_pvt import get_net
# from model.Denoiser_new import get_net
from util import AverageMeter
import cv2
from model.AgNet.core.utils import get_model, dice_loss
from opts import get_args


# def get_args():
#
#     parser = OptionParser()
#     parser.add_option('--data_path', dest='data_path',type='string',
#                       default='data/samples', help='data path')
#     parser.add_option('--model_path', dest='model_path',type='string',
#                       default='checkpoints/', help='model_path')
#     parser.add_option('--denoiser_path', dest='denoiser_path', type='string',
#                       default='checkpoints/denoiser/', help='denoiser_path')
#     parser.add_option('--classes', dest='classes', default=2, type='int',
#                       help='number of classes')
#     parser.add_option('--channels', dest='channels', default=3, type='int',
#                       help='number of channels')
#     parser.add_option('--width', dest='width', default=256, type='int',
#                       help='image width')
#     parser.add_option('--height', dest='height', default=256, type='int',
#                       help='image height')
#     parser.add_option('--model', dest='model', type='string',default='',
#                       help='model name(UNet, SegNet, DenseNet)')
#     parser.add_option('--batch_size', dest='batch_size', default=10, type='int',
#                       help='batch size')
#     parser.add_option('--adv_model', dest='adv_model', type='string',default='',
#                       help='model name(UNet, SegNet, DenseNet)')
#     parser.add_option('--GroupNorm', action="store_true", default=True,
#                         help='decide to use the GroupNorm')
#     parser.add_option('--BatchNorm', action="store_false", default=False,
#                         help='decide to use the BatchNorm')
#     parser.add_option('--data_type', dest='data_type', type='string',default='',
#                       help='org or DAG')
#     parser.add_option('--mode', dest='mode', type='string',default='test',
#                       help='mode test origin or adversarial')
#     parser.add_option('--gpu', dest='gpu',type='string',
#                       default='gpu', help='gpu or cpu')
#     parser.add_option('--attacks', dest='attacks', type='string', default="",
#                       help='attack types: Rician, DAG_A, DAG_B, DAG_C')
#     parser.add_option('--target', dest='target', default='', type='string',
#                       help='target class')
#     parser.add_option('--device1', dest='device1', default=0, type='int',
#                       help='device1 index number')
#     parser.add_option('--device2', dest='device2', default=-1, type='int',
#                       help='device2 index number')
#     parser.add_option('--device3', dest='device3', default=-1, type='int',
#                       help='device3 index number')
#     parser.add_option('--device4', dest='device4', default=-1, type='int',
#                       help='device4 index number')
#     parser.add_option('--output_path', dest='output_path', type='string',
#                       default='./output', help='output_path')
#     parser.add_option('--denoise_output', dest='denoise_output', type='string',
#                       default='./output/denoised_imgs/', help='denoise_output')
#
#     (options, args) = parser.parse_args()
#     return options


def test(model, denoiser, args):
    
    data_path = args.data_path
    n_classes = args.classes
    suffix = args.suffix
    # args.output_path = '{}/{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.adv_model, args.attacks)
    # args.output_path = os.path.join(args.output_path, args.data_path,'512', args.model, args.adv_model,
    #                                 args.data_type, args.attacks)
    # args.output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
    #                                 args.data_type, args.attacks, 'm'+ args.mask_type+'t'+args.target, suffix)
    # args.denoise_output = os.path.join(args.denoise_output, args.data_path, args.model, args.adv_model,
    #                                 args.data_type, args.attacks, 'm'+ args.mask_type+'t'+args.target, suffix)
    args.output_path = '{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.attacks)
    args.denoise_output = os.path.join(args.denoise_output, args.data_path, args.model,
                                    args.data_type, args.attacks, suffix)
    print('output path', args.output_path)
    print('denoised image path', args.denoise_output)
    device = torch.device(args.device)
        
    model = model.to(device)
    denoiser = denoiser.to(device)
    
    # set testdataset
        
    # test_dataset = SampleDataset(data_path,args.classes, args.channels, args.mode, None, args.adv_model, args.attacks,
    #                              args.target, args.data_type, args.width, args.height, args.mask_type, suffix)
    if args.attacks=='scl_attk':
        test_dataset = DefenseSclTestDataset(data_path, 'test', args.channels)
    else:
        test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)

    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))

    score = AverageMeter()
    
    # test
    
    model.eval()   # Set model to evaluate mode
    denoiser.eval()
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
        
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            if args.attacks == 'scl_attk':
                target = make_one_hot(labels, n_classes, device)
            else:
                target = make_one_hot(labels[:, 0, :, :], n_classes, device)
            denoised_img = denoiser(inputs)
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
                output.save(os.path.join(args.output_path,'{}.png'.format(batch_idx*args.batch_size+i)))
            denoised_img = np.moveaxis(denoised_img.cpu().numpy(),1, -1)
            for i, denoised_im in enumerate(denoised_img):
                # print(denoised_im.shape)
                cv2.imwrite(os.path.join(args.denoise_output,'{}.png'.format(batch_idx*args.batch_size+i)), denoised_im*255)
            del inputs, labels, target,  loss
            
    avg_score = score.avg
    if args.model == 'AG_Net':
        avg_score = 1-avg_score
    
    print('dice_score : {:.4f}'.format(avg_score))

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
        models_list = ['AG_Net']
        model_name = models_list[0]
        model = get_model(model_name)
        model = model(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
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
    prefix = 'pvt_scl_plus_leff'
    guide_mode = 'UNet'
    # guide_mode = 'DenseNet'
    denoiser_path = os.path.join(args.denoiser_path, args.data_path, '{}_{}.pth'.format(guide_mode,prefix))
    # denoiser_path = os.path.join(args.denoiser_path, args.data_path, '{}.pth'.format(guide_mode))
    print('denoiser ', denoiser_path)
    # denoiser = get_net(args.height, args.width, args.classes, args.channels, denoiser_path, args.batch_size)
    denoiser = get_net(args.height, args.channels, denoiser_path)
    # model_path = os.path.join(args.model_path, 'fundus', args.model + '.pth')
    print('target model', model_path)
    model.load_state_dict(torch.load(model_path))
    # summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

    test(model, denoiser, args)