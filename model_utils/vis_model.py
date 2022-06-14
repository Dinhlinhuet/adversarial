from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,\
    XGradCAM, EigenCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# from model.contrast_trans_localization2 import SupConTrans
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
from torch.utils.data.sampler import SequentialSampler
import cv2
import os
import shutil
import torch
import torch.nn as nn
import numpy as np

from opts import get_args

args = get_args()



device = torch.device(args.device)
bcz = args.batch_size
prefix = args.suffix
# prefix = 'pvt_scl_plus_leff_sub'
guide_mode = 'UNet'
# guide_mode = 'DenseNet'
denoiser_path = os.path.join(args.denoiser_path, args.data_path, args.attacks, '{}_{}.pth'.format(guide_mode, prefix))
# denoiser_path = os.path.join(args.denoiser_path, args.data_path, '{}.pth'.format(guide_mode))
print('denoiser ', denoiser_path)
# denoiser = get_net(args.height, args.width, args.classes, args.channels, denoiser_path, args.batch_size)
denoiser = get_net(args.height, args.channels, denoiser_path, args.middle)
# ckpt = torch.load(args.model_path, map_location='cpu')
# state_dict = ckpt['model']
# model.load_state_dict(state_dict)

def reshape_transform(tensor, height=11, width=11):
    # return tensor
    print("tenso", tensor.shape)
    # result = tensor.reshape(tensor.size(0), tensor.shape[-1],
    #                         height, width)
    # result = tensor.reshape(tensor.size(0),
    #                                   height, width, tensor.size(1)*tensor.size(2))
    height = int(np.sqrt(tensor.size(1)))
    width = height
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(-1))

    # # Bring the channels to the first dimension,
    # # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [[denoiser.middle_blocks[3]]]
data_path = args.data_path
attack = args.attacks
n_classes = args.classes
suffix = args.suffix
if args.attacks == 'scl_attk':
    test_dataset = DefenseSclTestDataset(data_path, 'test', args.channels, data_type=args.data_type, args=args)
elif any(attack in args.attacks for attack in ['dag', 'ifgsm']):
    if args.data_path == 'brain':
        test_dataset = AgDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                 args.attacks, args.data_type, args.mask_type, args.target, args.width, args.height,
                                 )
    elif args.data_path == 'lung':
        test_dataset = CovidDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                    args.attacks, args.data_type, args.mask_type, args.target, args.width,
                                    args.height,
                                    )
    else:
        test_dataset = SampleDataset(data_path, n_classes, args.channels, args.mode, args.model, \
                                     args.attacks, args.data_type, args.mask_type, args.target, args.width, args.height,
                                     )
else:
    test_dataset = AttackDataset(args.data_path, args.channels, args.mode, args.data_path)
test_sampler = SequentialSampler(np.arange(len(test_dataset)))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=bcz, sampler=test_sampler,
    num_workers=4, pin_memory=True)

# Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.gpus, device=device)
# cam = GradCAMPlusPlus(model=denoiser, target_layers=target_layers, use_cuda=args.gpus, device=device)
cam = GradCAMPlusPlus(model=denoiser, target_layers=target_layers, use_cuda=args.gpus, device=device,
                      reshape_transform=reshape_transform)
# cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=args.gpus)
# cam = FullGrad(model=model, target_layers=target_layers, use_cuda=args.gpus)
# cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=args.gpus)
# cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=args.gpus)

print('var', dir(cam))
# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
# root_dir =  './output/cam/section/early_layers_gradplus_class/{}/'.format(data_version)
# prefix = 'dec-patch_embed3.norm'
# prefix = 'dec-patch_embed'
prefix = 'mid-middle_blocks3'
root_dir = './output/cam/{}/{}/'.format(prefix, data_path, attack)
mam_dir = os.path.join(root_dir,'mam')
out_dir = '{}'.format(root_dir)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
if not os.path.exists(mam_dir):
    os.makedirs(mam_dir)

avg_imgs = []
# targets = [ClassifierOutputTarget()] * args.batch_size
for k, sample in enumerate(test_loader):
    imgs, labels = sample[0],sample[1]
    bcz = labels.shape[0]
    labels = labels.to(device)
    imgs = imgs.float()
    print('input', imgs.shape)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=imgs, labels=labels,
                                        targets=None)
    print("gray", grayscale_cam.shape)

    rgb_imgs = imgs.detach().cpu().numpy()
    print('rgb', rgb_imgs.shape)
    rgb_imgs = np.transpose(rgb_imgs, (0, 2, 3, 1))
    # # In this example grayscale_cam has only one image in the batch:
    for i, gray_cam in enumerate(grayscale_cam):
        # print("i", i)
        # print('imgi', imgs[i].shape)
        visualization = show_cam_on_image(rgb_imgs[i], gray_cam)
        cv2.imwrite('{}/{}.png'.format(out_dir, bcz*k+i), visualization)
        heatmap = cv2.applyColorMap(np.uint8(255 * gray_cam), cv2.COLORMAP_JET)
        avg_imgs.append(heatmap)

avg_imgs = np.mean(avg_imgs, 0)
# visualization = show_cam_on_image(rgb_imgs[i], avg_vessle)
cv2.imwrite('{}/mean_{}.png'.format(mam_dir,suffix), avg_imgs)
