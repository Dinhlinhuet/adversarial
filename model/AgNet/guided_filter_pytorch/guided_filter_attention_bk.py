from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from .box_filter import BoxFilter

class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12
        self.upsamples = [Upsample(512,512)
        , Upsample(256,256)
        , Upsample(128,128)
        , Upsample(64,64)]
        self.upsamples = torch.nn.ModuleList(self.upsamples)

    def forward(self, lr_x, lr_y, hr_x, l_a, idx):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()
        # lr_x = lr_x.float()
        # lr_y = lr_y.float()
        # hr_x = hr_x.float()
        # l_a = l_a.float()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        # print('A', A.size())
        # print('hrx w', h_hrx, w_hrx)
        ## mean_A; mean_b
        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')
        # mean_A = self.upsamples[idx](A,output_size=(h_hrx, w_hrx))
        # mean_b = self.upsamples[idx](b,output_size=(h_hrx, w_hrx))
        # return (mean_A * hr_x + mean_b)
        return (mean_A*hr_x+mean_b).double()

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x, output_size):
        return self.upsample(x, output_size)