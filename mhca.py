"""
2023, Multimodal Multi-Head Convolutional Attention with Various Kernel Sizes for Medical Image Super-Resolution
Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Andreea-Iuliana Miron, Olivian Savencu, Nicolae-Catalin Ristea, Nicolae Verga and Fahad Shahbaz Khan, WACV
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
"""

import torch.nn as nn


class MHCA(nn.Module):
    def __init__(self, n_feats, ratio):
        """
        MHCA spatial-channel attention module.
        :param n_feats: The number of filter of the input.
        :param ratio: Channel reduction ratio.
        """
        super(MHCA, self).__init__()

        out_channels = int(n_feats // ratio)

        head_1 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
        ]

        kernel_size_sam = 3
        head_2 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam, padding=0, bias=True)
        ]

        kernel_size_sam_2 = 5
        head_3 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=kernel_size_sam_2, padding=0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=kernel_size_sam_2, padding=0, bias=True)
        ]

        self.head_1 = nn.Sequential(*head_1)
        self.head_2 = nn.Sequential(*head_2)
        self.head_3 = nn.Sequential(*head_3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_h1 = self.head_1(x)
        res_h2 = self.head_2(x)
        res_h3 = self.head_3(x)
        m_c = self.sigmoid(res_h1 + res_h2 + res_h3)
        res = x * m_c
        return res
