import torch
from dataset.dataset import SampleDataset, AgDataset
from util import make_one_hot
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import cv2
import os

def cw_attack(args,dataset,model, kappa, adv_dir, device, targeted):
    def f(x, labels):

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)
    test_dataset = AgDataset(args.data_path, args.n_class, args.channels, args.mode, args.adv_model, args.attacks,
                             args.target,
                             args.data_type, args.img_size, args.img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )
    ws = torch.empty([0])
    model = model.double()
    learning_rate = 0.01
    m = 100
    b = 1e-3
    attack_imgs = []
    for i, (org_imgs, labels) in enumerate(test_loader):
        w = Variable(torch.atanh(org_imgs.clone() * 2 - 1), requires_grad=True)
        # w = torch.zeros_like(image, requires_grad=True).to(device)
        # w = torch.nn.Parameter(image, requires_grad=True)
        optimizer = optim.Adam([w], lr=learning_rate)
        pbar = tqdm(range(m), leave=False)
        for k in pbar:
            images = 1 / 2 * (nn.Tanh()(ws[i]) + 1)
            images = images.to(device).double()
            optimizer.zero_grad()
            mse = nn.MSELoss(reduction='sum')(images, org_imgs)
            loss1 = f(images, labels)
            cost = mse * b + loss1
            cost.requires_grad_()
            cost.backward()
            optimizer.step()
            pbar.set_description('%2.2f clean dice %f' % (((k + 1) / m * 100), loss1))
        attack_img = 1 / 2 * (nn.Tanh()(w) + 1)
        attack_imgs.append(attack_img)
        np_img = attack_img[0].detach().cpu().numpy()
        np_img = np.moveaxis(np_img, 0, -1)
        cv2.imwrite(os.path.join(adv_dir, '{}.png'.format(i)), np_img * 255)
