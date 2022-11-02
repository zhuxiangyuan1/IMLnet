import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from model.my_afnb import AFNB
from model.my_apnb import APNB
from model.module_helper import ModuleHelper
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class GSA1(nn.Module):
    def __init__(self,n_feats):
        super(GSA1, self).__init__()
        self.n_feats=n_feats
        self.conv11_head = conv3x3(2*n_feats, n_feats)
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 * self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )

        self.RB11 = nn.ModuleList()
        for i in range(16):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=1))

        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 *  self.n_feats, self.n_feats)
        )

        # condense layer
        self.condense = conv3x3(self.n_feats, self.n_feats)

    def forward(self, x, x1):
        f_ref = x
        cor = torch.cat([f_ref, x1], dim=1)
        cor = self.conv11_head(cor)
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)

        for i in range(16):
            cor = self.RB11[i](cor)

        cor = self.F_p(cor)
        cor1 = self.condense(w * cor)

        return cor1

class PASSRnet(nn.Module):
    def __init__(self, upscale_factor):
        super(PASSRnet, self).__init__()
        ### feature extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )

        self.fusion = AFNB(64, 64, 64, 64, 64, dropout=0.05, sizes=([1]),
                           norm_type='batchnorm')#sync_batchnorm
        self.conv_merge = conv3x3(64 * 2, 64)

        # self.GSA = GSA1(64)

        # extra added layers
        # self.context = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     ModuleHelper.BNReLU(64, norm_type='batchnorm'),
        #     APNB(in_channels=64, out_channels=64, key_channels=64, value_channels=64,
        #                  dropout=0.05, sizes=([1]), norm_type='batchnorm')
        # )
        # self.conv_merge2 = conv3x3(64 * 2, 64)

        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_right,is_training):
        ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)

        buffer = self.fusion(buffer_right,buffer_left)#buffer_right,buffer_left
        buffer=self.conv_merge(torch.cat((buffer_left, buffer), dim=1))
        # buffer=self.GSA(buffer_left, buffer)

        # buffer = self.context(buffer)
        # buffer = self.conv_merge2(torch.cat((buffer_left, buffer), dim=1))

        out = self.upscale(buffer)
        return out


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3