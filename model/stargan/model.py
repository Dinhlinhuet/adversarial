import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, in_chan=3,conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        enc_layers = []
        enc_layers.append(nn.Conv2d(in_chan+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        enc_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        enc_layers.append(nn.ReLU(inplace=True))

        # Down-sampling enc_layers.
        curr_dim = conv_dim
        for i in range(2):
            enc_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            enc_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            enc_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck enc_layers.
        for i in range(repeat_num):
            enc_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        dec_layers = []
        # Up-sampling layers.
        for i in range(2):
            dec_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            dec_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            dec_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        dec_layers.append(nn.Conv2d(curr_dim, in_chan, kernel_size=7, stride=1, padding=3, bias=False))
        dec_layers.append(nn.Tanh())
        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        out_enc = self.encoding(x,c)
        out = self.decoding(out_enc)
        return out

    def encoding(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        out = self.enc(x)
        return out

    def decoding(self, feat):
        out = self.dec(feat)
        return out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,in_chan, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_chan, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
