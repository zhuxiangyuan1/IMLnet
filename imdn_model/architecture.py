from model import lbnet_common as common
import torch.nn as nn
from . import block as B
from . import block_RFDN as B_RFDN
import torch
import numpy as np
from skimage import morphology
from model.my_afnb import AFNB
from emanet_zxy.EMANET_EMAU import EMAU
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class FDAM(nn.Module):
    def __init__(self, n_feats=32):
        super(FDAM, self).__init__()

        self.c1 = common.default_conv(n_feats, n_feats, 1)
        self.c2 = common.default_conv(n_feats, n_feats // 2, 3)
        self.c3 = common.default_conv(n_feats, n_feats // 2, 3)
        self.c4 = common.default_conv(n_feats*2, n_feats, 3)
        self.c5 = common.default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = common.default_conv(n_feats*2, n_feats, 1)

        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SpatialAttention()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        cat1 = torch.cat([y1, y2, y3], 1)
        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)  # 16
        cat2 = torch.cat([y2, y5, y4], 1)
        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)
        y6 = ca_out + sa_out
        y7 = self.c6(y6)
        output = res + y7

        return output


class IML(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        self.c = hdim
        self.k = kdim
        self.moving_average_rate = moving_average_rate
        self.units = nn.Embedding(kdim, hdim)

    def update(self, x, score, m=None):
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot  # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data

    def forward(self, x, update_flag=True):
        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)  # (n, c)
        m = self.units.weight.data  # (k, c)

        xn = F.normalize(x, dim=1)  # (n, c)
        mn = F.normalize(m, dim=1)  # (k, c)
        score = torch.matmul(xn, mn.t())  # (n, k)
        for i in range(3):
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1)  # (k, c)
            score = torch.matmul(xn, mn.t())  # (n, k)
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m)  # (n, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return out


class NETWORK(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(NETWORK, self).__init__()

        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, kernel_size=3),FDAM(n_feats=nf),FDAM(n_feats=nf))
        self.pam = PAM(64)
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, left,right):
        left = self.fea_conv(left)
        right=self.fea_conv(right)

        # out_fea=self.conv_merge(torch.cat((out_fea,right), dim=1))
        out_r_to_l,out_l_to_r=self.pam(left,right)

        out_B1_r_to_l = self.IMDB1(out_r_to_l)
        out_B1_l_to_r = self.IMDB1(out_l_to_r)

        out_B2_r_to_l = self.IMDB2(out_B1_r_to_l)
        out_B2_l_to_r = self.IMDB2(out_B1_l_to_r)

        out_B3_r_to_l = self.IMDB3(out_B2_r_to_l)
        out_B3_l_to_r = self.IMDB3(out_B2_l_to_r)

        out_B4_r_to_l = self.IMDB4(out_B3_r_to_l)
        out_B4_l_to_r = self.IMDB4(out_B3_l_to_r)

        out_B5_r_to_l = self.IMDB5(out_B4_r_to_l)
        out_B5_l_to_r = self.IMDB5(out_B4_l_to_r)

        out_B6_r_to_l = self.IMDB6(out_B5_r_to_l)
        out_B6_l_to_r = self.IMDB6(out_B5_l_to_r)

        out_B_r_to_l = self.c(torch.cat([out_B1_r_to_l, out_B2_r_to_l, out_B3_r_to_l, out_B4_r_to_l, out_B5_r_to_l, out_B6_r_to_l], dim=1))
        out_B_l_to_r = self.c(torch.cat([out_B1_l_to_r, out_B2_l_to_r, out_B3_l_to_r, out_B4_l_to_r, out_B5_l_to_r, out_B6_l_to_r], dim=1))

        out_lr_r_to_l = self.LR_conv(out_B_r_to_l) + out_r_to_l
        out_lr_l_to_r = self.LR_conv(out_B_l_to_r) + out_l_to_r

        output_r_to_l = self.upsampler(out_lr_r_to_l)
        output_l_to_r = self.upsampler(out_lr_l_to_r)

        return output_r_to_l,output_l_to_r

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


class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""

    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.iml = IML(64, 256, 0.999)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)##    torch.Size([16, 64, 30, 90])
        x = x * out + self.iml(x) ##  torch.Size([16, 64, 30, 90])
        return x

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2+1 , channels, 1, 1, 0, bias=True)
        self.afnb_zxy = AFNB(64, 64, 64, 64, 64, dropout=0.05, sizes=([1]),
                          norm_type='batchnorm')  # sync_batchnorm

        self.csa = CSAM_Module(channels)
        self.fusion_p_g = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)


    def __call__(self, x_left, x_right):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)#torch.Size([32, 64, 30, 90])

        afnb_right_to_left=self.afnb_zxy(buffer_right, buffer_left)
        afnb_left_to_right = self.afnb_zxy(buffer_left,buffer_right)

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W

        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))               # (B*H) * W * W  torch.Size([960, 90, 90])

        M_right_to_left = self.softmax(score)#右图到左图的视差注意力图

        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                # (B*H) * W * W torch.Size([960, 90, 90])

        M_left_to_right = self.softmax(score)#左图到右图的视差注意力图

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)


        #################################新加的##################################
        V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        V_right_to_left = V_right_to_left.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_right_to_left = morphologic_process(V_right_to_left)
        ###################################新加的#################################

        ### fusion
        buffer_r_to_l = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer_r_to_l = torch.bmm(M_right_to_left, buffer_r_to_l).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out_r_to_l = self.fusion(torch.cat((buffer_r_to_l, x_left,V_left_to_right), 1))

        ###################################新加的#################################
        buffer_l_to_r = self.b3(x_left).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer_l_to_r = torch.bmm(M_left_to_right, buffer_l_to_r).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out_l_to_r = self.fusion(torch.cat((buffer_l_to_r, x_right,V_right_to_left), 1))
        ####################################新加的################################

        out_l=self.fusion_p_g(torch.cat((out_r_to_l, afnb_right_to_left), 1))
        out_r = self.fusion_p_g(torch.cat((out_l_to_r, afnb_left_to_right), 1))

        out_l=self.csa(out_l)
        out_r = self.csa(out_r)

        return out_l,out_r


def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)
