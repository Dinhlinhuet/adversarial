import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from loss import combined_loss
from model.AgNet.core.utils import dice_loss
from pytorch_msssim import ssim
from util import make_one_hot

def i_fgsm(idx, model, n_class, x, y, y_target, targeted=False, eps=0.03, alpha=1, iteration=20, x_val_min=-1, x_val_max=1,
           background_class=0,device='cuda:0',verbose=False):
    alpha = eps/iteration
    x_adv = Variable(x.data, requires_grad=True)
    # x_adv = x
    # x_adv.require_grad()
    # y=y.long()
    y_target=y_target.long()
    # print('ytar', y_target.size())
    # loss_func = nn.CrossEntropyLoss()
    loss_func = combined_loss
    # loss_func = dice_loss
    # loss_func = nn.NLLLoss()
    # print('xadv', x_adv.size())
    # org = x_adv.clone().detach()
    # print('sub', torch.max(abs((org * 255) - (org * 255).int())))
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    for i in range(iteration):
        # print('i',i)
        #normal model
        h_adv = model(x_adv)
        # print('xadv', x_adv.size())
        # out, side_5, side_6, side_7, side_8 = model(x_adv)
        # output = torch.log(softmax_2d(side_8) + EPS)
        # # print('output', output.size())
        # _, h_adv = torch.max(output, 1)
        # h_adv = h_adv.float()

        # print("hadv", h_adv.size())

        # if targeted:
        #     cost = loss_func(h_adv, y)
        # else:
        #     cost = -loss_func(h_adv, y)
        # cost, _, _ = loss_func(h_adv, y, device, 2)
        cost, _, _ = loss_func(h_adv, y_target, device, n_class)
        # output = torch.argmax(output, 1)
        # print('uot', out.size())
        # h_adv = make_one_hot(output, n_class, device)

        # cost = loss_func(output, y_target)
        if not targeted:
            cost= -cost

        # print('xadv', x_adv.size())
        model.zero_grad()
        # print('xadv', x_adv)
        if x_adv.grad is not None:
            print('not none')
            x_adv.grad.data.fill_(0)
        cost.backward()
        # print('xadv', torch.unique(x_adv))
        x_adv.grad.sign_()
        add = alpha * x_adv.grad
        # print('equa', torch.all(torch.eq(org, x_adv)))
        # bef = x_adv.clone().detach()
        x_adv = x_adv - add
        # print('equa1', torch.all(torch.eq((bef*255).int(), (x_adv*255).int())))
        # print('sub', torch.unique((org * 255).int() - (x_adv * 255).int()))
        # print('sub', torch.max(abs((bef * 255) - (bef * 255).int())))
        # print('alpha', torch.min(abs(add)), torch.unique((add*255).int()))
        # org = x_adv.clone().detach()
        # print('add', torch.unique((add*255).int()))
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        # print('equax', torch.all(torch.eq(org,x_adv)))
        # tmp = x_adv.clone().detach()*255
        # print('tmp', torch.min(tmp), torch.max(tmp))
        # print('equa0', torch.unique((org * 255).int() - (x_adv * 255).int()))
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        # print('xadv', torch.unique((x_adv*255).int().detach()))
        # print('equa', torch.all(torch.eq(org, x_adv*255)))
        x_adv = Variable(x_adv.data, requires_grad=True)
        # print('equa', torch.max(abs((org * 255).int() - (x_adv * 255).int())))
        # final = x_adv.clone().detach()
        # print('final', torch.min(final), torch.max(final))
        # ssim_val = ssim(x, x_adv, data_range=255, size_average=False)

    return x_adv.detach().cpu().numpy()

def pgd(idx, model, n_class, x, y, y_target, targeted=False, eps=0.03, alpha=1, iteration=20, x_val_min=0, x_val_max=1,
           background_class=0,device='cuda:0',verbose=False):
    alpha = eps/iteration
    u=torch.FloatTensor(x.size()).uniform_(-eps, eps).to(device)
    x_adv = x + u
    x_adv = torch.clamp(x_adv, 0, 1)  # ensure valid pixel range
    x_adv = Variable(x_adv, requires_grad=True)
    # x_adv = x
    # x_adv.require_grad()
    # y=y.long()
    y_target=y_target.long()
    # loss_func = nn.CrossEntropyLoss()
    loss_func = combined_loss
    # print('xadv', x_adv.size())
    # org = x_adv.clone().detach()
    # print('sub', torch.max(abs((org * 255) - (org * 255).int())))
    for i in range(iteration):
        # print('i',i)
        h_adv = model(x_adv)
        # if targeted:
        #     cost = loss_func(h_adv, y)
        # else:
        #     cost = -loss_func(h_adv, y)
        # cost, _, _ = loss_func(h_adv, y, device, 2)
        cost, _, _ = loss_func(h_adv, y_target, device, n_class)
        if not targeted:
            cost= -cost
        # print('xadv', x_adv.size())
        model.zero_grad()
        # print('xadv', x_adv)
        if x_adv.grad is not None:
            print('not none')
            x_adv.grad.data.fill_(0)
        cost.backward()
        # print('xadv', torch.unique(x_adv))
        x_adv.grad.sign_()
        add = alpha * x_adv.grad
        # print('equa', torch.all(torch.eq(org, x_adv)))
        # bef = x_adv.clone().detach()
        x_adv = x_adv - add
        # print('equa1', torch.all(torch.eq((bef*255).int(), (x_adv*255).int())))
        # print('sub', torch.unique((org * 255).int() - (x_adv * 255).int()))
        # print('sub', torch.max(abs((bef * 255) - (bef * 255).int())))
        # print('alpha', torch.min(abs(add)), torch.unique((add*255).int()))
        # org = x_adv.clone().detach()
        # print('add', torch.unique((add*255).int()))
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        # print('equax', torch.all(torch.eq(org,x_adv)))
        # tmp = x_adv.clone().detach()*255
        # print('tmp', torch.min(tmp), torch.max(tmp))
        # print('equa0', torch.unique((org * 255).int() - (x_adv * 255).int()))
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        # print('xadv', torch.unique((x_adv*255).int().detach()))
        # print('equa', torch.all(torch.eq(org, x_adv*255)))
        x_adv = Variable(x_adv.data, requires_grad=True)
        # print('equa', torch.max(abs((org * 255).int() - (x_adv * 255).int())))
        # final = x_adv.clone().detach()
        # print('final', torch.min(final), torch.max(final))
        # ssim_val = ssim(x, x_adv, data_range=255, size_average=False)

    return x_adv.detach().cpu().numpy()

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)