'''
Function for Dense Adversarial Generation
Adversarial Examples for Semantic Segmentation
Muhammad Ferjad Naeem
ferjad.naeem@tum.de
adapted from https://github.com/IFL-CAMP/dense_adversarial_generation_pytorch
'''
import torch
from util import make_one_hot
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
# from loss import dice_loss
from model.AgNet.core.utils import dice_loss as ag_dice_loss
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

def DAG(args, model,image,ground_truth,adv_target,num_iterations=20,gamma=0.07,no_background=True,
        background_class=0,device='cuda:0',verbose=False):
    '''
    Generates adversarial example for a given Image
    
    Parameters
    ----------
        model: Torch Model
        image: Torch tensor of dtype=float. Requires gradient. [b*c*h*w]
        ground_truth: Torch tensor of labels as one hot vector per class
        adv_target: Torch tensor of dtype=float. This is the purturbed labels. [b*classes*h*w]
        num_iterations: Number of iterations for the algorithm
        gamma: epsilon value. The maximum Change possible.
        no_background: If True, does not purturb the background class
        background_class: The index of the background class. Used to filter background
        device: Device to perform the computations on
        verbose: Bool. If true, prints the amount of change and the number of values changed in each iteration
    Returns
    -------
        Image:  Adversarial Output, logits of original image as torch tensor
        logits: Output of the Clean Image as torch tensor
        noise_total: List of total noise added per iteration as numpy array
        noise_iteration: List of noise added per iteration as numpy array
        prediction_iteration: List of prediction per iteration as numpy array
        image_iteration: List of image per iteration as numpy array

    '''
    # print('input', image.size())
    noise_total=[]
    noise_iteration=[]
    prediction_iteration=[]
    image_iteration=[]
    background=None
    if args.model!='AgNet':
        logits=model(image)
    else:
        out, side_5, side_6, side_7, side_8 = model(image)
        softmax_2d = nn.Softmax2d()
        EPS = 1e-12
        logits = torch.log(softmax_2d(side_8) + EPS)
    # out = F.upsample(out, size=(image.size()[2], image.size()[3]), mode='bilinear')
    # sigmoid = nn.Sigmoid()
    #normal model
    # orig_image=image
    # _,predictions_orig=torch.max(logits,1)
    # print('logit', torch.unique(predictions_orig))
    # predictions_orig=make_one_hot(predictions_orig,logits.shape[1],device)
    
    if(no_background):
        # print('logitshap', logits.shape)
        background=torch.zeros(logits.shape)
        background[:,background_class,:,:]=torch.ones((background.shape[2],background.shape[3]))
        background=background.to(device)

    if torch.all(torch.eq(ground_truth, adv_target)):
        print('equally')
        return image
    org_img = image.clone()
    dices = []
    for a in range(num_iterations):
        if args.model != 'AgNet':
            output=model(image)
        else:
            out, side_5, side_6, side_7, side_8 = model(image)
            output = torch.log(softmax_2d(side_8) + EPS)
        # print('output', output.size())
        _,predictions=torch.max(output,1)
        prediction_iteration.append(predictions[0].cpu().numpy())
        # print('img iter', image.size())
        # predictions=make_one_hot(predictions,logits.shape[1],device)
        # print('pre', predictions.size(), ground_truth.size())
        #select correct pixels Tn
        # condition1=torch.eq(predictions,ground_truth)
        # condition=condition1
       
        # if no_background:
        #     # condition2=(ground_truth!=background)
        #     condition2 = (adv_target != background)
        #     condition=torch.mul(condition1,condition2)
        # condition=condition.float()
        # # print('condi', condition.size())
        # if(condition.sum()==0):
        #     print("Condition Reached")
        #     if a==0:
        #         np_img = image[0].detach().cpu().numpy()
        #         np_img = np.moveaxis(np_img, 0, -1)
        #         image_iteration.append(np_img)
        #         return _, _, _, _, _, image_iteration
        #     image=None
        #     break

        # #Finding pixels to purturb
        # adv_log=torch.mul(output,adv_target)
        # #Getting the values of the original output
        # clean_log=torch.mul(output,ground_truth)

        # adv_dice = dice_loss(output, adv_target)
        clean_dice = dice_loss(output, ground_truth)
        # r_f = adv_dice - clean_dice
        # r_f = adv_dice
        r_f = clean_dice
        dices.append(clean_dice.mean())
        # r_m = adv_dice - clean_dice
        # print("rf", r_f)
        print("cld", clean_dice)
        #Finding r_m
        # adv_direction=adv_log-clean_log
        # r_f=torch.mul(adv_direction,condition)

        # ms_ssim_val = ssim(image, org_img , data_range=1, size_average=False)
        # # mse = nn.MSELoss(reduction='mean')(image, org_img)
        # # print('msssim', mse)
        # r_m = r_f + ms_ssim_val
        # # r_m = r_f
        # # print("rm", r_m.size())
        # r_m.requires_grad_()
        # r_m_grad = torch.autograd.grad(r_m, image,grad_outputs=torch.zeros_like(r_m), retain_graph=True)
        # # Saving gradient for calculation
        # # r_m_norm = r_m_grad[0]
        # # print('sub',np.all(torch.eq(r_m,0).cpu().numpy()))
        # # r_m_adv=torch.mul(adv_log,condition)
        # # r_m_adv.requires_grad_()
        # #
        # # r_m_clean=torch.mul(clean_log,condition)
        # # r_m_clean.requires_grad_()
        #
        # # grad_outputs = torch.ones(r_m.size()).cuda()
        # # ri_grad_0 = torch.autograd.grad(r_m, image, grad_outputs=grad_outputs, retain_graph=True)
        # # ri_grad_adv = torch.autograd.grad(r_m_adv, image, grad_outputs=grad_outputs, retain_graph=True)
        # # ri_grad_clean = torch.autograd.grad(r_m_clean, image, grad_outputs=grad_outputs, retain_graph=True)
        # # cv2.imwrite('./output/debug/gradient/adv_log_{}.png'.format(idx), ri_grad_adv[0].cpu().numpy())
        # # cv2.imwrite('./output/debug/gradient/cl_log_{}.png'.format(idx), ri_grad_clean[0].cpu().numpy())
        # # adv =ri_grad_adv[0].cpu().numpy()
        # # clean = ri_grad_clean[0].cpu().numpy()
        # # print('minmax', np.amin(adv), np.amax(adv),np.amin(clean), np.amax(clean))
        # # print('rm', np.count_nonzero(ri_grad_0[0].cpu().numpy()))
        # # print('pr',torch.count_nonzero(ri_grad_0[0]).cpu().numpy())
        # # print('out', torch.equal(ri_grad_adv[0],ri_grad_clean[0]), torch.count_nonzero(ri_grad_adv[0]-ri_grad_clean[0])
        # #       .cpu().numpy())
        # #Summation
        # # r_m_sum=r_m.sum()
        # # r_m_sum.requires_grad_()
        # # print('rm0')
        # # print('rms', r_m_sum.size())
        # #Finding gradient with respect to image
        # # r_m_grad=torch.autograd.grad(r_m_sum,image,retain_graph=True)
        # # r_m_grad = torch.autograd.grad(r_m, image, retain_graph=True)
        # #Saving gradient for calculation
        # # print('rmgrad', len(r_m_grad), r_m_grad[0].size())
        # r_m_grad_calc=r_m_grad[0]
        # # r_m_grad_calc = r_m_grad
        # # print('rmgrad', np.count_nonzero(r_m_grad_calc.cpu().numpy()))
        # #Calculating Magnitude of the gradient
        # r_m_grad_mag=r_m_grad_calc.norm()
        #
        # # if(r_m_grad_mag==0):
        # #     print("Condition Reached, no gradient")
        # #     #image=None
        # #     np_img = image[0].detach().cpu().numpy()
        # #     np_img = np.moveaxis(np_img, 0, -1)
        # #     image_iteration.append(np_img)
        # #     return _, _, _, _, _, image_iteration
        # #     break
        # #Calculating final value of r_m
        # r_m_norm=(gamma/r_m_grad_mag)*r_m_grad_calc
        # # print('rmn', r_m_norm.size())
        # # print("r_m_norm_ : ", torch.unique(r_m_norm))
        # #if no_background:
        # #if False:
        #
        # # if no_background is False:
        # #     condition_image=condition.sum(dim=1)
        # #     condition_image=condition_image.unsqueeze(1)
        # #     r_m_norm=torch.mul(r_m_norm,condition_image)
        #
        # #Updating the image
        # # print("r_m_norm : ",torch.min(r_m_norm))
        # # print("bef norm", torch.min(r_m_norm), torch.max(r_m_norm))
        # # r_m_norm = clip_grad(r_m_norm, image)
        # # image = image+r_m_norm
        # # image=torch.clamp((image+r_m_norm),0,1)
        # # image= sigmoid(image+r_m_norm)
        print('clm', image.size())
        # print("val", torch.min(r_m_norm), torch.max(r_m_norm), r_m_norm.size())
        # print("valimg", torch.min(image), torch.max(image))
        # np_img = image[0][0].detach().cpu().numpy()
        # image_iteration.append(np_img)
        # noise_total.append((image-orig_image)[0][0].detach().cpu().numpy())
        # noise_iteration.append(r_m_norm[0][0].cpu().numpy())

        # if verbose:
        #     print("Iteration ",a)
        #     print("Change to the image is ",r_m_norm.sum())
        #     print("Magnitude of grad is ",r_m_grad_mag)
        #     print("Condition 1 ",condition1.sum())
        #     if no_background:
        #         print("Condition 2 ",condition2.sum())
        #         print("Condition is", condition.sum())\
    avg_dice = torch.mean(dices)
    print('avg', avg_dice)
    np_img = image.detach().cpu().numpy()
    # print('np1', np_img)
    np_img = np.moveaxis(np_img, 0, -1)

    return  np_img


def cw(args, idx, model, image, ground_truth, oh_label, adv_target, num_iterations=20, gamma=0.07, no_background=True,
        background_class=0, device='cuda:0', verbose=False):
    '''
    Generates adversarial example for a given Image

    Parameters
    ----------
        model: Torch Model
        image: Torch tensor of dtype=float. Requires gradient. [b*c*h*w]
        ground_truth: Torch tensor of labels as one hot vector per class
        adv_target: Torch tensor of dtype=float. This is the purturbed labels. [b*classes*h*w]
        num_iterations: Number of iterations for the algorithm
        gamma: epsilon value. The maximum Change possible.
        no_background: If True, does not purturb the background class
        background_class: The index of the background class. Used to filter background
        device: Device to perform the computations on
        verbose: Bool. If true, prints the amount of change and the number of values changed in each iteration
    Returns
    -------
        Image:  Adversarial Output, logits of original image as torch tensor
        logits: Output of the Clean Image as torch tensor
        noise_total: List of total noise added per iteration as numpy array
        noise_iteration: List of noise added per iteration as numpy array
        prediction_iteration: List of prediction per iteration as numpy array
        image_iteration: List of image per iteration as numpy array

    '''
    print('input', image.size())
    prediction_iteration = []
    image_iteration = []
    background = None
    if args.model != 'AgNet':
        logits = model(image)
    else:
        out, side_5, side_6, side_7, side_8 = model(image)
        softmax_2d = nn.Softmax2d()
        EPS = 1e-12
        logits = torch.log(softmax_2d(side_8) + EPS)
    # out = F.upsample(out, size=(image.size()[2], image.size()[3]), mode='bilinear')
    # sigmoid = nn.Sigmoid()
    # normal model
    # orig_image=image
    # _,predictions_orig=torch.max(logits,1)
    # print('logit', torch.unique(predictions_orig))
    # predictions_orig=make_one_hot(predictions_orig,logits.shape[1],device)

    # if (no_background):
    #     # print('logitshap', logits.shape)
    #     background = torch.zeros(logits.shape)
    #     background[:, background_class, :, :] = torch.ones((background.shape[2], background.shape[3]))
    #     background = background.to(device)

    if torch.all(torch.eq(oh_label, adv_target)):
        # print('equally')
        # np_img = image[0].detach().cpu().numpy()
        # np_img = np.moveaxis(np_img, 0, -1)
        np_img = image.detach().cpu().numpy()
        # np_img = np.moveaxis(np_img, 1, -1)
        return np_img
    # org_img = image.clone()
    kappa = 0.0
    prev = 1e10
    learning_rate = 0.1
    img_area = image.size()[2]*image.size()[-1]
    # img_area = 1
    print('area', img_area)
    # w = torch.Tensor(image, requires_grad=True).to(device)
    w = Variable(torch.atanh(image.clone()*2-1), requires_grad=True)
    # w = torch.zeros_like(image, requires_grad=True).to(device)
    # w = torch.nn.Parameter(image, requires_grad=True)
    optimizer = optim.Adam([w], lr=learning_rate)
    c= 1e1
    pbar = tqdm(range(num_iterations))
    for step in pbar:
        img = 1 / 2 * (nn.Tanh()(w) + 1)
        if args.model != 'AgNet':
            output = model(img)
            # _, predictions = torch.max(output, 1)
        else:
            out, side_5, side_6, side_7, side_8 = model(img)
            output = torch.log(softmax_2d(side_8) + EPS)
        _, predictions = torch.max(output, 1)
        predictions=make_one_hot(predictions,logits.shape[1],device)
        # print('output', output.size())
        prediction_iteration.append(predictions[0].cpu().numpy())
        # print('pre', predictions.size(), ground_truth.size())
        # select correct pixels Tn
        condition1=torch.eq(predictions,oh_label)
        condition=condition1

        if no_background:
            # condition2=(ground_truth!=background)
            condition2 = (adv_target != background)
            condition=torch.mul(condition1,condition2)
        condition=condition.float()
        # print('condi', condition.size())
        if(condition.sum()==0):
            print("Condition Reached")
            # if a==0:
                # np_img = image[0].detach().cpu().numpy()
                # np_img = np.moveaxis(np_img, 0, -1)
            np_img = image.detach().cpu().numpy()
            # np_img = np.moveaxis(np_img, 1, -1)
            return np_img
            # break

        # #Finding pixels to purturb
        # adv_log=torch.mul(output,adv_target)
        # #Getting the values of the original output
        # clean_log=torch.mul(output,ground_truth)
        # if args.model == 'AgNet':
        #     adv_dice = ag_dice_loss(predictions, adv_target)
        #     clean_dice = ag_dice_loss(predictions, ground_truth)
        # else:
        #     adv_dice = dice_loss(output, adv_target)
        #     clean_dice = dice_loss(output, ground_truth)
        # r_f = adv_dice - clean_dice
        # print('output', output.size())
        adv_log = torch.mul(output, adv_target)
        # Getting the values of the original output
        clean_log = torch.mul(output, oh_label)
        adv_direction = adv_log - clean_log
        # r_m = torch.mul(adv_direction, condition)
        r_m = adv_direction
        r_m_sum = r_m.sum()/img_area
        # print("rmsum", r_m_sum)
        # Finding r_m
        # adv_direction=adv_log-clean_log
        # r_f=torch.mul(adv_direction,condition)

        ms_ssim_val = ssim(img, image, data_range=1, size_average=False).mean()
        cost = 1 - ms_ssim_val + torch.abs(r_m_sum) * c
        # mse = nn.MSELoss(reduction='sum')(img, image)
        # cost = mse + r_m_sum * c
        # print('mse', ms_ssim_val)
        # Saving gradient for calculation
        # r_m_norm = r_m_grad[0]
        # print('sub',np.all(torch.eq(r_m,0).cpu().numpy()))
        # r_m_adv=torch.mul(adv_log,condition)
        # r_m_adv.requires_grad_()
        #
        # r_m_clean=torch.mul(clean_log,condition)
        # r_m_clean.requires_grad_()

        # grad_outputs = torch.ones(r_m.size()).cuda()
        # ri_grad_0 = torch.autograd.grad(r_m, image, grad_outputs=grad_outputs, retain_graph=True)
        # ri_grad_adv = torch.autograd.grad(r_m_adv, image, grad_outputs=grad_outputs, retain_graph=True)
        # ri_grad_clean = torch.autograd.grad(r_m_clean, image, grad_outputs=grad_outputs, retain_graph=True)
        # cv2.imwrite('./output/debug/gradient/adv_log_{}.png'.format(idx), ri_grad_adv[0].cpu().numpy())
        # cv2.imwrite('./output/debug/gradient/cl_log_{}.png'.format(idx), ri_grad_clean[0].cpu().numpy())
        # adv =ri_grad_adv[0].cpu().numpy()
        # clean = ri_grad_clean[0].cpu().numpy()
        # print('minmax', np.amin(adv), np.amax(adv),np.amin(clean), np.amax(clean))
        # print('rm', np.count_nonzero(ri_grad_0[0].cpu().numpy()))
        # print('pr',torch.count_nonzero(ri_grad_0[0]).cpu().numpy())
        # print('out', torch.equal(ri_grad_adv[0],ri_grad_clean[0]), torch.count_nonzero(ri_grad_adv[0]-ri_grad_clean[0])
        #       .cpu().numpy())
        # Summation
        # r_m_sum=r_m.sum()
        # r_m_sum.requires_grad_()
        # print('rm0')
        # print('rms', r_m_sum.size())
        # Finding gradient with respect to image
        # r_m_grad=torch.autograd.grad(r_m_sum,image,retain_graph=True)
        # r_m_grad = torch.autograd.grad(r_m, image, retain_graph=True)
        # Saving gradient for calculation
        # r_m_grad_calc = r_m_grad[0]
        # print('rmgrad', np.count_nonzero(r_m_grad_calc.cpu().numpy()))
        # Calculating Magnitude of the gradient
        # r_m_grad_mag = r_m_grad_calc.norm()

        # if(r_m_grad_mag==0):
        #     print("Condition Reached, no gradient")
        #     #image=None
        #     np_img = image[0].detach().cpu().numpy()
        #     np_img = np.moveaxis(np_img, 0, -1)
        #     image_iteration.append(np_img)
        #     return _, _, _, _, _, image_iteration
        #     break
        # Calculating final value of r_m
        # r_m_norm = (gamma / r_m_grad_mag) * r_m_grad_calc
        # print('rmn', r_m_norm.size())
        # print("r_m_norm_ : ", torch.unique(r_m_norm))
        # if no_background:
        # if False:

        # if no_background is False:
        #     condition_image=condition.sum(dim=1)
        #     condition_image=condition_image.unsqueeze(1)
        #     r_m_norm=torch.mul(r_m_norm,condition_image)

        # Updating the image
        # print("r_m_norm : ",torch.min(r_m_norm))
        # print("bef norm", torch.min(r_m_norm), torch.max(r_m_norm))
        # r_m_norm = clip_grad(r_m_norm, image)
        # image = image+r_m_norm
        # image = torch.clamp((image + r_m_norm), 0, 1)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # image= sigmoid(image+r_m_norm)
        # print('clm', image.size())
        # print("val", torch.min(r_m_norm), torch.max(r_m_norm), r_m_norm.size())
        # print("valimg", torch.min(img), torch.max(img))
        # np_img = image[0][0].detach().cpu().numpy()
        # np_img = img[0].detach().cpu().numpy()
        # np_img = np.moveaxis(np_img, 0, -1)
        np_img = img.detach().cpu().numpy()
        # np_img = np.moveaxis(np_img, 1, -1)
        image_iteration.append(np_img)
        if step % (num_iterations // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                break
            prev = cost
        # print('- Learning Progress : %2.2f %%        ' % ((step + 1) / num_iterations * 100), cost, ms_ssim_val, r_m_sum, end='\r')
        # pbar.set_description('- Learning Progress : %2.2f cost %f ssim %f r_sum %f ' % (((step + 1) / num_iterations * 100),
        #                                                                                 cost, mse, r_m_sum))
        pbar.set_description(
            '- Learning Progress : %2.2f cost %f ssim %f r_sum %f ' % (((step + 1) / num_iterations * 100),
                                                                       cost, ms_ssim_val, r_m_sum))
        # noise_total.append((image-orig_image)[0][0].detach().cpu().numpy())
        # noise_iteration.append(r_m_norm[0][0].cpu().numpy())

        # if verbose:
        #     print("Iteration ",a)
        #     print("Change to the image is ",r_m_norm.sum())
        #     print("Magnitude of grad is ",r_m_grad_mag)
        #     print("Condition 1 ",condition1.sum())
        #     if no_background:
        #         print("Condition 2 ",condition2.sum())
        #         print("Condition is", condition.sum())
    img = 1 / 2 * (nn.Tanh()(w) + 1)
    print('output', img.size())
    return img.detach().cpu().numpy()

def clip_grad(grad, img):
    max = 1 - img
    if 0 in max:
        k = 0
    else:
        norm = grad/max
        k = min(norm[norm>0])
    grad = grad*k
    min = img
    if 0 in min:
        l = 0
    else:
        norm = grad / min
        l = torch.max(norm[norm<0])
    grad = grad*l
    return grad


def dice_loss(pred, encoded_target):
    """
    :param pred : N x C x H x W logits
    :param encoded_target : N x C x H x W LongTensor
    """

    output = F.softmax(pred, dim=1)
    eps = 1
    # print('out', output.size())
    intersection = output * encoded_target
    numerator = 2 * intersection.sum(1).sum(1) + eps
    denominator = output + encoded_target
    denominator = denominator.sum(1).sum(1) + eps
    # print('deno', denominator)
    loss_per_channel = 1 - (numerator / denominator)
    # print('chn', loss_per_channel.size())
    # print('out', output.size(1))
    loss = loss_per_channel.sum(dim=1) / output.size(1)
    del output, encoded_target
    # print('loss', loss.size())
    return loss