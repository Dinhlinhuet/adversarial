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

def DAG(args, idx, model,image,ground_truth,adv_target,num_iterations=20,gamma=0.07,no_background=True,
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
    print('input', image.size())
    prediction_iteration=[]
    image_iteration=[]
    background=None
    if args.model != 'AgNet':
        logits = model(image)
    else:
        out, side_5, side_6, side_7, side_8 = model(image)
        softmax_2d = nn.Softmax2d()
        EPS = 1e-12
        logits = torch.log(softmax_2d(side_8) + EPS)
    # out = F.upsample(out, size=(image.size()[2], image.size()[3]), mode='bilinear')

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
        # print('equally')
        return image
    for a in range(num_iterations):
        if args.model != 'AgNet':
            output = model(image)
        else:
            out, side_5, side_6, side_7, side_8 = model(image)
            output = torch.log(softmax_2d(side_8) + EPS)
        # print('output', output.size())
        _,predictions=torch.max(output,1)
        prediction_iteration.append(predictions[0].cpu().numpy())
        predictions=make_one_hot(predictions,logits.shape[1],device)
        # print('pre', predictions.size(), ground_truth.size())
        #select correct pixels Tn
        condition1=torch.eq(predictions,ground_truth)
        condition=condition1
       
        if no_background:
            # condition2=(ground_truth!=background)
            condition2 = (adv_target != background)
            condition=torch.mul(condition1,condition2)
        condition=condition.float()
        # print('condi', condition.size())
        if(condition.sum()==0):
            print("Condition Reached")
            return image
            break

        #Finding pixels to purturb
        adv_log=torch.mul(output,adv_target)
        #Getting the values of the original output
        clean_log=torch.mul(output,ground_truth)

        #Finding r_m
        adv_direction=adv_log-clean_log
        r_m=torch.mul(adv_direction,condition)
        r_m.requires_grad_()

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
        #Summation
        r_m_sum=r_m.sum()
        r_m_sum.requires_grad_()
        # print('rm0', r_m_sum)
        # print('rms', r_m_sum.size())
        #Finding gradient with respect to image
        r_m_grad=torch.autograd.grad(r_m_sum,image,grad_outputs=torch.zeros_like(r_m_sum), retain_graph=True)
        #Saving gradient for calculation
        r_m_grad_calc=r_m_grad[0]
        # print('rmgrad', np.count_nonzero(r_m_grad_calc.cpu().numpy()))
        #Calculating Magnitude of the gradient
        r_m_grad_mag=r_m_grad_calc.norm()
        
        if(r_m_grad_mag==0):
            print("Condition Reached, no gradient",a)
            return image
            break
        #Calculating final value of r_m
        r_m_norm=(gamma/r_m_grad_mag)*r_m_grad_calc
        # print('rmn', r_m_norm.size())
        # print("r_m_norm : ", torch.unique(r_m_norm))
        #if no_background:
        #if False:
        if no_background is False:
            condition_image=condition.sum(dim=1)
            condition_image=condition_image.unsqueeze(1)
            r_m_norm=torch.mul(r_m_norm,condition_image)

        #Updating the image
        # print("r_m_norm : ",torch.min(r_m_norm))
        image=torch.clamp((image+r_m_norm),0,1)
        # print('clm', image.size())
        # np_img = image[0][0].detach().cpu().numpy()
    np_img = image
    return np_img