import os
import torch
import numpy as np
import scipy.misc as smp
import scipy.ndimage
from random import randint
import random
import cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_target_bs(idx, y_test, target_class=1, width=256, height=256):
    y_target = y_test
    # print('yt', y_target.size(), y_target[0].size())
    # print('tar', target_class)
    # print('ytarget bef', torch.sum(y_target), torch.unique(y_target))
    y_target[y_target == target_class] = 0
    # print('ytarget af', torch.sum(y_target), torch.unique(y_target))
    # npimg = np.uint8((y_target * (255/2)).cpu().numpy())
    # print('ytg', npimg.shape)
    # cv2.imwrite('./output/debug/ytarget/fundus/dilated_{}.png'.format(idx),npimg[0])
    return y_target

def generate_target_swap_cls(idx, y_test, target_class=1, width=256, height=256):
    y_target = y_test
    # print('yt', y_target.size(), y_target[0].size())
    # print('tar', target_class)
    # print('ytarget bef', torch.sum(y_target), torch.unique(y_target))
    y_target[y_target == target_class] = 1-target_class
    y_target[y_target == 1-target_class] = target_class
    # print('ytarget af', torch.sum(y_target), torch.unique(y_target))
    # npimg = np.uint8((y_target * (255/2)).cpu().numpy())
    # print('ytg', npimg.shape)
    # cv2.imwrite('./output/debug/ytarget/fundus/dilated_{}.png'.format(idx),npimg[0])
    return y_target

def generate_target(idx, y_test, target_class = 13, width = 256, height = 256):
    
    y_target = y_test

    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=12).astype(y_test.dtype)
    # print('dilate', np.unique(dilated_image))
    cv2.imwrite('./output/debug/ytarget/octa/dilated_{}.png'.format(idx), dilated_image*255)
    # print('dilate', dilated_image.shape)
    # print('ytg', y_target.shape)
    # print('tg', y_target[0, target_class, :, :].shape)
    # for i in range(width):
    #     for j in range(height):
    #         y_target[0, target_class, i, j] = dilated_image[i,j]
    y_target[0, target_class, :, :] = dilated_image
    # temp = y_target.copy()
    for i in range(width):
        for j in range(height):
            potato = np.count_nonzero(y_target[0,:,i,j])
            # print('potato', potato)
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                # print(x)
                # print('k',k)
                #print("{}, {}, {}".format(i,j,k))
                if k[0] == target_class:
                    #if background is target
                    y_target[0,k[1],i,j] = 0.
                else:
                    #if other class is target
                    y_target[0, k[0], i, j] = 0.
    # print('check', np.all(temp==y_target))
    # print('check1', np.all(y_test == y_target))
    return y_target

def generate_target_swap(y_test):


    y_target = y_test
    # print('ytest', y_test.shape)
    y_target_arg = np.argmax(y_test, axis = 1)
    # print('arg', y_target_arg.shape)
    y_target_arg_no_back = np.where(y_target_arg>0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes  = np.unique(y_target_arg)
    org_classes = np.arange(y_target.shape[1])
    # print("classes", classes)
    swapped= True
    # if len(classes) > 3:
    if len(classes) > 1:
        first_class = 0

        second_class = 0

        # third_class = 0

        while first_class == second_class:
            first_class = classes[randint(0, len(classes)-1)]
            # f_ind = np.where(y_target_arg==first_class)
            # print(np.shape(f_ind))

            # second_class = classes[randint(0, len(classes)-1)]
            second_class = org_classes[randint(0, len(org_classes)-1)]
            # s_ind = np.where(y_target_arg == second_class)

            # third_class = classes[randint(0, len(classes) - 1)]
            # t_ind = np.where(y_target_arg == third_class)

        # summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]
        # summ = np.shape(f_ind)[1] + np.shape(s_ind)[1]

        for i in range(256):
            for j in range(256):
                temp = y_target[0,second_class, i,j]
                y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
                y_target[0, first_class,i, j] = temp


    else:
        y_target = y_test
        swapped=False
        print('Not enough classes to swap!')
    y_target = np.argmax(y_target, axis = 1)
    return y_target, swapped

# def generate_target_swap(y_test):
#
#
#     y_target = y_test
#     # print('ytest', y_test.shape)
#     y_target_arg = np.argmax(y_test, axis = 1)
#     # print('arg', y_target_arg.shape)
#     y_target_arg_no_back = np.where(y_target_arg>0)
#
#     y_target_arg = y_target_arg[y_target_arg_no_back]
#
#     classes  = np.unique(y_target_arg)
#     # print("classes", classes)
#     swapped= True
#     # if len(classes) > 3:
#     if len(classes) > 1:
#         first_class = 0
#
#         second_class = 0
#
#         third_class = 0
#
#         while first_class == second_class == third_class:
#         # while first_class == second_class:
#             first_class = classes[randint(0, len(classes)-1)]
#             f_ind = np.where(y_target_arg==first_class)
#             # print(np.shape(f_ind))
#
#             second_class = classes[randint(0, len(classes)-1)]
#             s_ind = np.where(y_target_arg == second_class)
#
#             third_class = classes[randint(0, len(classes) - 1)]
#             t_ind = np.where(y_target_arg == third_class)
#
#             summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]
#             # summ = np.shape(f_ind)[1] + np.shape(s_ind)[1]
#
#             if summ < 1000:
#                 first_class = 0
#
#                 second_class = 0
#
#                 third_class = 0
#
#         for i in range(256):
#             for j in range(256):
#                 temp = y_target[0,second_class, i,j]
#                 y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
#                 y_target[0, first_class,i, j] = temp
#
#
#     else:
#         y_target = y_test
#         swapped=False
#         print('Not enough classes to swap!')
#     return y_target, swapped
