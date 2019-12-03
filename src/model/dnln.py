from model import common
import torch
import torch.nn as nn
from model.common import ResBlock
from torch.nn import functional as F
import numpy as np

from Deform_Conv.modules.modulated_deform_conv import ModulatedDeformConv_Sep as DeformConv
from model.RRDBNet import RRDBNet
from model.HFFB import HFFB


def make_model(args):
    return DNLN(args)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, supp_feature, ref_feature):
        x = supp_feature  # b,c,h,w
        y = ref_feature

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)

        g_y = g_y.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_y)

        f_div_C = F.softmax(f, dim=1)

        x1 = torch.matmul(f_div_C, g_y)

        x1 = x1.permute(0, 2, 1).contiguous()

        x1 = x1.view(batch_size, self.inter_channels, *supp_feature.size()[2:])
        W_x1 = self.W(x1)
        z = x + W_x1

        return z


class DNLN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DNLN, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        n_frames = args.n_frames
        self.n_deform_conv = args.n_deform_conv
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Feature Extraction Module #
        self.feat0 = conv(args.n_colors, n_feats, kernel_size)

        modules_body1 = [
            ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=self.act, res_scale=1) for _ in
            range(5)]
        modules_body1.append(conv(n_feats, n_feats, kernel_size))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Alignment Module #
        self.fuse_layers = nn.ModuleList()
        self.HFFBs = nn.ModuleList()
        self.dconvs = nn.ModuleList()

        for i in range(self.n_deform_conv):
            self.fuse_layers.append(nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True))
            self.HFFBs.append(HFFB(n_feats))
            self.dconvs.append(DeformConv(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=1,
                                          dilation=1,
                                          groups=1,
                                          deformable_groups=8, im2col_step=1))

        # Non-local Attention Module #
        self.non_local = NonLocalBlock(n_feats, n_feats // 2)

        # Reconstruction Module #
        self.res_feat2 = RRDBNet(n_feats * n_frames, n_feats, nb=23, gc=32)

        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def align(self, flames_feature):
        feature = []
        batch_size, num, ch, h, w = flames_feature.size()
        ref_feature = flames_feature[:, num // 2, :, :, :].clone()
        for i in range(num):
            if i == num // 2:
                feature.append(ref_feature)
                continue
            supp_feature = flames_feature[:, i, :, :, :].contiguous()

            for j in range(self.n_deform_conv):
                fusion = self.act(self.fuse_layers[j](torch.cat((supp_feature, ref_feature), 1)))
                fusion = self.HFFBs[j](fusion)
                supp_feature = self.dconvs[j](supp_feature, fusion)

            fea = self.non_local(supp_feature, ref_feature)

            feature.append(fea)
        return feature

    def forward(self, x):
        batch_size, num, c, h, w = x.size()

        flames = x.view(-1, c, h, w)
        flames_feature = self.act(self.feat0(flames))
        flames_feature = self.res_feat1(flames_feature)
        flames_feature = flames_feature.view(batch_size, num, -1, h, w)
        a_feature = self.align(flames_feature)

        feature = torch.cat(a_feature, dim=1)

        res = self.res_feat2(feature)
        res += a_feature[num // 2]

        output = self.tail(res)

        return output
