import os
import numpy as np
import argparse
from torch.backends import cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader,Sampler
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))
from dataset.gan_dataset import GanDataset
from model.stargan.solver import Solver


def str2bool(v):
    return v.lower() in ('true')

def main(args):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Data loader.
    data_path = args.data_path
    channels = args.channels
    train_dataset = GanDataset(data_path, channels, mode= 'train',width=args.image_size,height=args.image_size)
    # val_dataset = AgDataset(data_path, n_class,3, 'val', None, None, None, 'org', args.img_size, args.img_size)
    # Solver for training and testing StarGAN.
    train_sampler = SubsetRandomSampler(np.arange(len(train_dataset)))
    # val_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        sampler=train_sampler
    )
    solver = Solver(train_loader, args)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--data_path', dest='data_path',type=str,
                      default='fundus', help='data path')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                      help='number of epochs')
    parser.add_argument('--dataset', type=str, default='fundus')
    parser.add_argument('--channels', dest='channels', default=3, type=int,
                      help='number of channels')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device', dest='device', default=0, type=int,
                      help='device1 index number')

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/fundus/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='./checkpoints/stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    args = parser.parse_args()
    print(args)
    main(args)