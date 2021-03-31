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

from util import make_one_hot
from dataset import SampleDataset
from model import UNet, SegNet, DenseNet
from loss import dice_score
from model.Denoiser import get_net
import cv2


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--model_path', dest='model_path',type='string',
                      default='checkpoints/', help='model_path')
    parser.add_option('--classes', dest='classes', default=2, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--model', dest='model', type='string',default='',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--batch_size', dest='batch_size', default=10, type='int',
                      help='batch size')
    parser.add_option('--adv_model', dest='adv_model', type='string',default='',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--data_type', dest='data_type', type='string',default='',
                      help='org or DAG')
    parser.add_option('--mode', dest='mode', type='string',default='test',
                      help='mode test origin or adversarial')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
    parser.add_option('--attacks', dest='attacks', type='string', default="",
                      help='attack types: Rician, DAG_A, DAG_B, DAG_C')
    parser.add_option('--target', dest='target', default='', type='string',
                      help='target class')
    parser.add_option('--device1', dest='device1', default=0, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')
    parser.add_option('--output_path', dest='output_path', type='string',
                      default='./output', help='output_path')
    parser.add_option('--denoise_output', dest='denoise_output', type='string',
                      default='./output/denoised_imgs/', help='denoise_output')

    (options, args) = parser.parse_args()
    return options


def test(model, args):
    
    data_path = args.data_path
    gpu = args.gpu
    n_classes = args.classes
    data_width = args.width
    data_height = args.height
    # args.output_path = '{}/{}/{}/{}/{}/'.format(args.output_path,args.data_path,args.model,args.adv_model, args.attacks)
    # args.output_path = os.path.join(args.output_path, args.data_path,'512', args.model, args.adv_model,
    #                                 args.data_type, args.attacks)
    args.output_path = os.path.join(args.output_path, args.data_path, args.model, args.adv_model,
                                    args.data_type, args.attacks, args.target)
    args.denoise_output = os.path.join(args.denoise_output, args.data_path, args.model, args.adv_model,
                                    args.data_type, args.attacks, args.target)
    print('output path', args.output_path)
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
    
    # set testdataset
        
    test_dataset = SampleDataset(data_path,args.classes, args.channels, args.mode, args.adv_model,
                                 args.attacks, args.target, args.data_type, args.width, args.height)
    
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
    if not os.path.exists(args.denoise_output):
        os.makedirs(args.denoise_output)
    # cm = plt.cm.jet
    cm = plt.get_cmap('gist_rainbow',1000)
    # cm= plt.get_cmap('viridis', 28)
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(test_loader):
        
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            
            target = make_one_hot(labels[:,0,:,:], n_classes, device)
            
            pred, denoised_img = model(device, inputs, train=False, defense=True)
            # pred = model(inputs, train=False, defense=False)
            # print('pred size', denoised_img.size())
            loss, masks = dice_score(pred,target)
            
            avg_score += loss.data.cpu().numpy()
            # print('af',len(pred.data.cpu().numpy()))
            masks=onehot2norm(np.asarray(masks.data.cpu()))
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
            # for i, denoised_im in enumerate(denoised_img.cpu().numpy()):
            #     # print(denoised_im.shape)
            #     cv2.imwrite(os.path.join(args.denoise_output,'{}.png'.format(batch_idx*args.batch_size+i)), denoised_im[0]*255)
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
    
    # model = None
    
    # if args.model == 'UNet':
    #     model = UNet(in_channels = n_channels, n_classes = n_classes)
    #
    # elif args.model == 'SegNet':
    #     model = SegNet(in_channels = n_channels, n_classes = n_classes)
    #
    # elif args.model == 'DenseNet':
    #     model = DenseNet(in_channels = n_channels, n_classes = n_classes)
    #
    # else :
    #     print("wrong model : must be UNet, SegNet, or DenseNet")
    #     raise SystemExit
        
    model_path = os.path.join(args.model_path, args.data_path, args.model + '.pth')
    model = get_net(args.height, args.width, args.classes, args.channels, None)
    model.load_state_dict(torch.load(model_path))
    summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
    
    test(model, args)