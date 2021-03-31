import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import random
import sys
import os
import cv2

from model import UNet, SegNet, DenseNet

from dataset.dataset import SampleDataset, AgDataset
from scipy.stats import rice
from skimage.measure import compare_ssim as ssim
from pytorch_msssim import ms_ssim, ssim
# from dag import DAG
from dag_iqa import DAG, cw
# from dag_ssim import DAG
from dag_utils import generate_target, generate_target_swap, generate_target_bs
from util import make_one_hot
from attack.fgsm import i_fgsm, pgd
from model.AgNet.core.models import AG_Net
from model.AgNet.core.utils import get_model

from optparse import OptionParser

BATCH_SIZE = 10


def get_args():
    
    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path',type='string',
                      default='data/samples', help='data path')
    parser.add_option('--attack_path', dest='attack_path',type='string',
                      default=None, help='the path of adversarial attack examples')
    parser.add_option('--model_path', dest='model_path',type='string', default='./checkpoints/',
                      help='model_path')
    parser.add_option('--classes', dest='classes', default=2, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--GroupNorm', action="store_true", default= True,
                        help='decide to use the GroupNorm')
    parser.add_option('--BatchNorm', action="store_false", default = False,
                        help='decide to use the BatchNorm')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--attacks', dest='attacks', type='string',
                      help='attack types: Rician, DAG_A, DAG_B, DAG_C')
    parser.add_option('--target', dest='target', default='0', type='string',
                      help='target class')
    parser.add_option('--gpu', dest='gpu',type='string',
                      default='gpu', help='gpu or cpu')
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

def load_data(args):
    
    data_path = args.data_path
    n_classes = args.classes

    # generate loader
    # test_dataset = AgDataset(data_path, n_classes, args.channels, 'adv',args.model,args.attacks, None,'org',
    #                              args.width, args.height)
    test_dataset = SampleDataset(data_path, n_classes, args.channels, 'adv', args.model, args.attacks, None, 'org',
                                 args.width, args.height)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
    
    print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    
    return test_dataset, test_loader

# generate Rician noise examples
# Meausre the difference between original and adversarial examples by using structural Similarity (SSIM). 
# The adversarial examples which has SSIM value from 0.97 to 0.99 can be passed.
# SSIM adapted from https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

def DAG_Attack(model, test_dataset, args):
    
    # Hyperparamter for DAG 
    
    num_iterations=1000
    # gamma=0.5
    gamma = 1
    num=15

    gpu = args.gpu
    
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

    target_class = int(args.target)
    adversarial_examples = []
    # adv_dir = './output/adv/{}/train/{}/{}/{}/'.format(args.data_path,args.model,args.attacks, target_class)
    # adv_dir = './output/adv/{}/val/{}/{}/{}/'.format(args.data_path, args.model, args.attacks, target_class)
    adv_dir = './output/adv/{}/{}/{}/{}/'.format(args.data_path, args.model, args.attacks,target_class)
    print('adv dir', adv_dir)
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)
    org_imgs, adv_imgs = [],[]
    for batch_idx in range(len(test_dataset)):
        image, label = test_dataset.__getitem__(batch_idx)
        # print('image', image.size())
        # for j, img in enumerate(image):
        #     print('shape',img.shape)
        org_imgs.append(image)
        image = image.unsqueeze(0)
        # pure_label = label.squeeze(0).numpy()
        # print('un', np.unique(pure_label))
        image , label = image.clone().detach().requires_grad_(True).float(), label.clone().detach().float()
        # image, label = image.clone().detach().requires_grad_(True).long(), label.clone().detach().long()
        image , label = image.to(device), label.to(device)
        # label = label*255
        label = label.long()
        # unique = torch.unique(label)
        # print('unique', unique.cpu().numpy())
        # for i, cl in enumerate(unique):
        #     label[label == cl] = i
        # Change labels from [batch_size, height, width] to [batch_size, num_classes, height, width]
        # label_oh=make_one_hot(label.long(),n_classes,device)
        label_oh = make_one_hot(label, n_classes, device)
        swapped=True
        print(args.attacks)
        if 'DAG_A' in args.attacks :
            print('dagA')
            adv_target = torch.zeros_like(label_oh)

        elif 'DAG_B' in args.attacks:
            print('dagB')
            adv_target,swapped=generate_target_swap(label_oh.cpu().numpy())
            adv_target=torch.from_numpy(adv_target).float()

        elif 'DAG_C' in args.attacks:
            print('dagC')
            # choice one randome particular class except background class(0)
            # unique_label = torch.unique(label)
            # target_class = int(random.choice(unique_label[1:]).item())
            # target_class = 2
            adv_target = generate_target_bs(batch_idx, label, target_class=target_class)
            # adv_target=generate_target(batch_idx, label_oh.cpu().numpy(), target_class = target_class)
            # print('checkout', np.all(adv_target==label_oh.cpu().numpy()))
            adv_target=make_one_hot(adv_target, n_classes, device)
        elif ('ifgsm' in args.attacks) or ('pgd' in args.attacks)  :
            print('ifgsm,pgd')
            # adv_target = torch.zeros_like(label)

            adv_target, swapped = generate_target_swap(label_oh.cpu().numpy())
            adv_target = torch.from_numpy(adv_target).float()
        else :
            print('else')
            print("wrong adversarial attack types : must be DAG_A, DAG_B, or DAG_C")
            raise SystemExit


        adv_target=adv_target.to(device)
        if 'ifgsm' in args.attacks:
            image_iteration = i_fgsm(idx=batch_idx, model=model,
                                     n_class=args.classes,
                                     x=image,
                                     y=label,
                                     y_target=adv_target,
                                     iteration=num_iterations,
                                     background_class=0,
                                     device=device,
                                     verbose=False)
            out_img = image_iteration[0] * 255
            out_img = np.moveaxis(out_img, 0, -1)
            print('img', batch_idx, image_iteration.shape, out_img.dtype)
            cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(batch_idx)), out_img)
            adv_imgs.append(image_iteration[0])
        elif 'pgd' in args.attacks:
            image_iteration = pgd(idx=batch_idx, model=model,
                                     n_class=args.classes,
                                     x=image,
                                     y=label,
                                     y_target=adv_target,
                                     iteration=num_iterations,
                                     background_class=0,
                                     device=device,
                                     verbose=False)
            out_img = image_iteration[0] * 255
            out_img = np.moveaxis(out_img, 0, -1)
            print('img', image_iteration.shape, out_img.dtype)
            cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(batch_idx)), out_img)
            adv_imgs.append(image_iteration[0])
        elif 'cw' in args.attacks:
            _, _, _, _, _, image_iteration = cw(args, idx=batch_idx, model=model,
                                                 image=image,
                                                 ground_truth=label_oh,
                                                 adv_target=adv_target,
                                                 num_iterations=num_iterations,
                                                 gamma=gamma,
                                                 no_background=False,
                                                 background_class=0,
                                                 device=device,
                                                 verbose=False)

            if len(image_iteration) > 0:
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(batch_idx)), image_iteration[-1] * 255)
            else:
                np_img = image[0].detach().cpu().numpy()
                np_img = np.moveaxis(np_img, 0, -1)
                cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(batch_idx)), np_img * 255)
            adv_imgs.append(image_iteration[-1])
        else:
            if swapped:
                _, _, _, _, _, image_iteration=DAG(args,idx= batch_idx, model=model,
                          image=image,
                          ground_truth=label_oh,
                          adv_target=adv_target,
                          num_iterations=num_iterations,
                          gamma=gamma,
                          no_background=False,
                          background_class=0,
                          device=device,
                          verbose=False)

                if len(image_iteration)>0:
                    cv2.imwrite(os.path.join(adv_dir,'{}.png'.format(batch_idx)),image_iteration[-1]*255)
                else:
                    np_img = image[0].detach().cpu().numpy()
                    np_img = np.moveaxis(np_img, 0, -1)
                    cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(batch_idx)), np_img* 255)
                adv_imgs.append(image_iteration[-1])

        # if len(image_iteration) >= 1:
        #
        #     adversarial_examples.append([image_iteration[-1],
        #                                  pure_label])
        #
        # del image_iteration
    org_imgs = torch.movedim(torch.stack(org_imgs),1,-1)
    adv_imgs = torch.Tensor(np.stack(adv_imgs))
    print('shape', org_imgs.size(), adv_imgs.size())
    ssim_val = ssim(org_imgs, adv_imgs, data_range=1, size_average=False)
    avg_ssim = torch.mean(ssim_val).numpy()
    ms_ssim_val = ms_ssim(org_imgs, adv_imgs, data_range=1, size_average=False)
    avg_msssim = torch.mean(ms_ssim_val).numpy()
    print('avg ssim: ', avg_ssim, 'avg msssim: ', avg_msssim)
    # print('total {} {} images are generated'.format(len(adversarial_examples), args.attacks))
    print('done generating')
    return adversarial_examples

if __name__ == "__main__":

    args = get_args()
    
    n_channels = args.channels
    n_classes = args.classes
    
    test_dataset, test_loader = load_data(args)
    
    if args.attacks == 'Rician':
        
        adversarial_examples = Rician(test_dataset)
        
        if args.attack_path is None:
            
            # adversarial_path = 'data/' + args.attacks + '.pickle'
            adversarial_path = 'data/' + args.attacks + '.pickle'
        else:
            
            adversarial_path = args.attack_path
        
    else:
        
        model = None

        if args.model == 'UNet':
            model = UNet(in_channels = n_channels, n_classes = n_classes)

        elif args.model == 'SegNet':
            model = SegNet(in_channels = n_channels, n_classes = n_classes)

        elif args.model == 'DenseNet':
            model = DenseNet(in_channels = n_channels, n_classes = n_classes)

        elif args.model == 'AgNet':
            model = AG_Net(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)
            # model = get_model('AG_Net')
            # model = model(n_classes=n_classes, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

        else :
            print("wrong model : must be UNet, SegNet, or DenseNet")
            raise SystemExit

        summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')

        model_path = os.path.join(args.model_path,args.data_path,args.model+'.pth')
        print('Load model', model_path)
        model.load_state_dict(torch.load(model_path))

        adversarial_examples = DAG_Attack(model, test_dataset, args)
        
        if args.attack_path is None:
            
            adversarial_path = 'data/' + args.model + '_' + args.attacks + '.pickle'
            
        else:
            adversarial_path = args.attack_path
        
    # save adversarial examples([adversarial examples, labels])
    # with open(adversarial_path, 'wb') as fp:
    #     pickle.dump(adversarial_examples, fp)
    
