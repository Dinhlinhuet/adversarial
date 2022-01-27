"""
Use a composite loss of weighted-cross entropy and dice loss proposed in https://arxiv.org/pdf/1801.04161.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os.path
import numpy as np
from util import estimate_weights, make_one_hot
from torch.autograd import Variable

def onehot2norm(imgs, device):
    out = torch.argmax(imgs,dim=1,keepdim=True).long()
    # print("out", out.size())
    one_hot = torch.FloatTensor(imgs.shape).zero_()
    one_hot= one_hot.to(device)
    one_hot.scatter_(1, out, 1)
    # out = Variable(out.data, requires_grad=True)
    one_hot = torch.tensor(one_hot, dtype=torch.float64, requires_grad=True)
    # print('out', out.size())
    return one_hot

def dice_score(pred, encoded_target):
    """
    :param pred : N x C x H x W logits
    :param encoded_target : N x C x H x W LongTensor
    """
    
    output = F.softmax(pred, dim = 1)
    # print('out', output)
    eps = 1
 
    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    
    loss_per_channel = numerator / denominator
    
    score = loss_per_channel.sum() / output.size(1)

    # del output, encoded_target
    del encoded_target

    return score.mean(), output


def dice_loss(pred, encoded_target):
    """
    :param pred : N x C x H x W logits
    :param encoded_target : N x C x H x W LongTensor
    """
    
    output = F.softmax(pred, dim = 1)
    eps = 1
    # print('out', output.size())
    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    
    loss_per_channel = 1 - (numerator / denominator)
    
    loss = loss_per_channel.sum() / output.size(1)
    del output, encoded_target
    
    return loss.mean()


def cross_entropy_loss(pred, target, weight):
    """
    :param pred : N x C x H x W
    :param target : N x H x W
    :param: weight : N x H x W
    
    """
    
    loss_func = nn.CrossEntropyLoss()
    
    loss = loss_func(pred, target)
        
    return torch.mean(torch.mul(loss, weight))

def combined_loss(pred, target, device, n_classes):
    """
    :param pred: N x C x H x W
    :param target: N x H x W
    """

    # target = target.float()
    weights = estimate_weights(target.float())
    weights = weights.to(device)
    # print('pred,target', pred.size(), target.size(), pred.dtype, weights.dtype)
    cross = cross_entropy_loss(pred, target.long(), weights)
    # print('combine')
    target_oh = make_one_hot(target, n_classes, device)
    dice = dice_loss(pred, target_oh)
    
    loss = cross + dice
    
    del weights
    
    return loss, cross, dice


def dice_loss1(pred, target,device, n_classes,):
    """
    :param pred : N x C x H x W logits
    :param encoded_target : N x C x H x W LongTensor
    """
    encoded_target = make_one_hot(target, n_classes, device)
    output = F.softmax(pred, dim=1)
    eps = 1
    # print('out', output.size())
    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + eps

    loss_per_channel = 1 - (numerator / denominator)

    loss = loss_per_channel.sum() / output.size(1)
    del output, encoded_target

    return loss.mean()