"""
Forked from https://github.com/sanghyun-son/EDSR-PyTorch.
MHCA code has the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) licence.
"""
import math

import torch
import torch.nn as nn


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, args=None):

        super(ResBlock, self).__init__()

        self.use_mhca_3 = args.use_attention_resblock and args.use_mhca_3
        self.use_mhca_2 = args.use_attention_resblock and args.use_mhca_2

        ratio = float(args.ratio) # mhca channel reduction ratio

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        # -- MHCA
        if self.use_mhca_2 or self.use_mhca_3:
            kernel_size_sam = 3
            out_channels = int(n_feats // ratio)
            spatial_attention = [
                nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0, bias=True)
            ]
            channel_attention = [
                nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
            ]

            self.spatial_attention = nn.Sequential(*spatial_attention)
            self.channel_attention = nn.Sequential(*channel_attention)
            self.sigmoid = nn.Sigmoid()

            kernel_size_sam_2 = 5
            spatial_attention_2 = [
                nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0, bias=True)
            ]
            self.spatial_attention_2 = nn.Sequential(*spatial_attention_2)
        # -- END MHCA

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        if self.use_mhca_2:
            channel = self.channel_attention(res)
            spatial = self.spatial_attention(res)
            """
            m_c = tf.nn.sigmoid(channel + spatial)
            x_tilde = tf.multiply(input_, m_c)
            """
            m_c = self.sigmoid(channel + spatial)
            res = res * m_c

        if self.use_mhca_3:
            channel = self.channel_attention(res)
            spatial = self.spatial_attention(res)
            spatial_2 = self.spatial_attention_2(res)
            m_c = self.sigmoid(channel + spatial + spatial_2)
            res = res * m_c


        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

