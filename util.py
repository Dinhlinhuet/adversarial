import numpy as np
import torch
import torch.nn as nn

def estimate_weights(labels):
    '''
    reference from https://github.com/ai-med/quickNAT_pytorch
    Estimate weights which balance the relative importance of pixesl in the LogitsticLoss
    more detailed in https://arxiv.org/pdf/1801.04161.pdf
    '''
    
    if torch.is_tensor(labels):
         labels = labels.cpu().numpy()
        
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    # print('label', labels.shape, unique, counts)
    median_freq = np.median(counts)
    
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
    
    # grads = np.gradient(labels)
    # print('grad', len(grads),grads[1].shape,grads[2].shape)
    # edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    # edge_weights = (grads[1] ** 2 + grads[2] ** 2) > 0
    # class_weights += 2 * edge_weights
    
    class_weights = torch.tensor(class_weights).float()
    
    return class_weights


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label to a one-hot values.

    Parameters
    ----------
        labels : N x H x W, where N is batch size.(torch.Tensor)
        num_classes : int
        device: torch.device information
    -------
    Returns
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    # print('sha', labels.size())
    labels=labels.unsqueeze(1)
    # print('label', torch.unique(labels))
    # print('lable', np.unique(labels.data.cpu().numpy()))
    # labels = torch.nn.functional.one_hot(labels,3)
    # print("label", labels.size())
    labels = labels.long()
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def save_metrics(metrics, size, loss=None, cross = None, dice = None):
    '''
    loss value save in metrics
    '''
    
    # if cross is not None:
    #     metrics['cross'] += cross.data.cpu().numpy() * size
    #
    # if dice is not None:
    #     metrics['dice'] += dice.data.cpu().numpy() * size
    #
    # metrics['loss'] += loss.data.cpu().numpy() * size
    if cross is not None:
        metrics['cross'] += cross.detach().data.cpu().numpy() * size

    if dice is not None:
        metrics['dice'] += dice.detach().data.cpu().numpy() * size
    if loss is not None:
        metrics['loss'] += loss.detach().data.cpu().numpy() * size


def save_denoise_metrics(metrics, size, loss=None, cross=None, dice=None):
    '''
    loss value save in metrics
    '''

    if cross is not None:
        metrics['cross'] += cross.data.cpu().numpy() * size

    if dice is not None:
        metrics['dice'] += dice.data.cpu().numpy() * size

    metrics['loss'] += loss.data.cpu().numpy() * size

def print_metrics(metrics, epoch_size, phase):    
    '''
    print metrics which saves loss value
    '''
    
    outputs = []
    values = []
    for k in metrics.keys():
        value = metrics[k] / epoch_size
        outputs.append("{}: {:4f}".format(k, value))
        values.append(value)
        
    print("{}: {}".format(phase, ", ".join(outputs)))
    return values

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count