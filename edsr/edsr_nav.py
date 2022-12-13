"""
Forked from https://github.com/sanghyun-son/EDSR-PyTorch.
Added MMHCA. This class is used for multi-input SR.
MHCA code has the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) licence.
"""

import common

import torch.nn as nn
import torch


def make_model(args, parent=False):
    return EDSR_Nav(args)


class EDSR_Nav(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_Nav, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        ratio = float(args.ratio)   # mhca channel reduction ratio

        self.shift_mean = args.shift_mean
        assert self.shift_mean is False
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head_n = [conv(args.n_colors, n_feats, kernel_size)]
        m_head_a = [conv(args.n_colors, n_feats, kernel_size)]
        m_head_v = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body_n = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, args=args,
            ) for _ in range(n_resblocks)
        ]
        m_body_n.append(conv(n_feats, n_feats, kernel_size))

        m_body_a = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, args=args
            ) for _ in range(n_resblocks)
        ]
        m_body_a.append(conv(n_feats, n_feats, kernel_size))


        m_body_v = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, args=args
            ) for _ in range(n_resblocks)
        ]
        m_body_v.append(conv(n_feats, n_feats, kernel_size))


        n_feats_nav = n_feats * 2
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats_nav, act=False),
            conv(n_feats_nav, args.n_colors, kernel_size)
        ]

        self.use_mhca_2 = args.use_mhca_2
        self.use_mhca_3 = args.use_mhca_3

        if self.use_mhca_2 or self.use_mhca_3:
            """
            spatial = tf.layers.conv2d(input_, filters=new_c, kernel_size=kernel_size_sam, strides=1,
                                       padding='VALID', activation=tf.nn.relu, reuse=reuse, name='conv2_spatial_down')
    
            spatial = tf.layers.conv2d_transpose(spatial, filters=C, kernel_size=kernel_size_sam, strides=1,
                                                 padding='VALID', activation=None, reuse=reuse, name='conv2_spatial_up')
            """

            self.sigmoid = nn.Sigmoid()
            kernel_size_sam = 3
            out_channels = int(n_feats_nav // ratio)
            spatial_attention = [
                nn.Conv2d(in_channels=n_feats_nav, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats_nav, kernel_size=kernel_size_sam, padding=0, bias=True)
            ]

            """
            channel = tf.layers.conv2d(input_, filters=new_c, kernel_size=1, strides=1,
                                       padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2_channel_down')
            channel = tf.layers.conv2d(channel, filters=C, kernel_size=1, strides=1,
                                       padding='SAME', activation=None, reuse=reuse, name='conv2_channel_up')
            """

            channel_attention = [
                nn.Conv2d(in_channels=n_feats_nav, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
                nn.ReLU(True),
                nn.Conv2d(in_channels=out_channels, out_channels=n_feats_nav, kernel_size=1, padding=0, bias=True)
            ]
            self.spatial_attention = nn.Sequential(*spatial_attention)
            self.channel_attention = nn.Sequential(*channel_attention)

            kernel_size_sam_2 = 5
            spatial_attention_2 = [
                nn.Conv2d(in_channels=n_feats_nav, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats_nav, kernel_size=kernel_size_sam_2, padding=0, bias=True)
            ]

            self.spatial_attention_2 = nn.Sequential(*spatial_attention_2)

        self.head_n = nn.Sequential(*m_head_n)
        self.head_a = nn.Sequential(*m_head_a)
        self.head_v = nn.Sequential(*m_head_v)

        self.body_n = nn.Sequential(*m_body_n)
        self.body_a = nn.Sequential(*m_body_a)
        self.body_v = nn.Sequential(*m_body_v)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, input_):
        # t2w, t1w, pd = input_
        t2w, pd = input_
        res_head_t2w = self.head_n(t2w)
        res_t2w = self.body_n(res_head_t2w) + res_head_t2w

        res_head_pd = self.head_a(pd)
        res_pd = self.body_a(res_head_pd) + res_head_pd

        res = torch.cat((res_t2w, res_pd), dim=1)

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
            """
            m_c = tf.nn.sigmoid(channel + spatial)
            x_tilde = tf.multiply(input_, m_c)
            """
            m_c = self.sigmoid(channel + spatial + spatial_2)
            res = res * m_c

        x = self.tail(res)

        if self.shift_mean:
            x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

