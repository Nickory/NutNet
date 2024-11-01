#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-19 17:20:11
#   Description : pytorch_yolov4
#
# ================================================================
import torch
import torch.nn.functional as F


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Conv2dUnit(torch.nn.Module):
    def __init__(self, input_dim, filters, kernels, stride=1, padding=0, bn=1, act='mish'):
        super(Conv2dUnit, self).__init__()
        use_bias = (bn != 1)
        self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=use_bias)
        self.bn = None
        if bn:
            self.bn = torch.nn.BatchNorm2d(filters)
        self.act = None
        if act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1, inplace = True)
        elif act == 'mish':
            self.act = Mish()

        # 参数初始化。不这么初始化，容易梯度爆炸nan
        # self.conv.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, input_dim, kernels[0], kernels[1])))
        # self.bn.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        # self.bn.bias.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        # self.bn.running_mean.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        # self.bn.running_var.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters_1, filters_2):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters_1, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters_1, filters_2, (3, 3), stride=1, padding=1)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class StackResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters_1, filters_2, n):
        super(StackResidualBlock, self).__init__()
        self.sequential = torch.nn.Sequential()
        for i in range(n):
            self.sequential.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters_1, filters_2))
    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x


class SPP(torch.nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        # self.conv = torch.nn.MaxPool2d(kernel_size, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, 1, 2)
        x_3 = F.max_pool2d(x, 9, 1, 4)
        x_4 = F.max_pool2d(x, 13, 1, 6)
        out = torch.cat([x_4, x_3, x_2, x_1], dim=1)
        return out



class YOLOv4(torch.nn.Module):
    def __init__(self, num_classes, num_anchors, initial_filters=32, use_gpu=False):
        super(YOLOv4, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_gpu = use_gpu
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        # cspdarknet53部分
        self.conv001 = Conv2dUnit(3, i32, 3, stride=1, padding=1)

        # ============================= s2 =============================
        self.conv002 = Conv2dUnit(i32, i64, 3, stride=2, padding=1)
        self.conv003 = Conv2dUnit(i64, i64, 1, stride=1)
        self.conv004 = Conv2dUnit(i64, i64, 1, stride=1)
        self.stackResidualBlock01 = StackResidualBlock(i64, i32, i64, n=1)
        self.conv007 = Conv2dUnit(i64, i64, 1, stride=1)
        self.conv008 = Conv2dUnit(i128, i64, 1, stride=1)

        # ============================= s4 =============================
        self.conv009 = Conv2dUnit(i64, i128, 3, stride=2, padding=1)
        self.conv010 = Conv2dUnit(i128, i64, 1, stride=1)
        self.conv011 = Conv2dUnit(i128, i64, 1, stride=1)
        self.stackResidualBlock02 = StackResidualBlock(i64, i64, i64, n=2)
        self.conv016 = Conv2dUnit(i64, i64, 1, stride=1)
        self.conv017 = Conv2dUnit(i128, i128, 1, stride=1)

        # ============================= s8 =============================
        self.conv018 = Conv2dUnit(i128, i256, 3, stride=2, padding=1)
        self.conv019 = Conv2dUnit(i256, i128, 1, stride=1)
        self.conv020 = Conv2dUnit(i256, i128, 1, stride=1)
        self.stackResidualBlock03 = StackResidualBlock(i128, i128, i128, n=8)
        self.conv037 = Conv2dUnit(i128, i128, 1, stride=1)
        self.conv038 = Conv2dUnit(i256, i256, 1, stride=1)

        # ============================= s16 =============================
        self.conv039 = Conv2dUnit(i256, i512, 3, stride=2, padding=1)
        self.conv040 = Conv2dUnit(i512, i256, 1, stride=1)
        self.conv041 = Conv2dUnit(i512, i256, 1, stride=1)
        self.stackResidualBlock04 = StackResidualBlock(i256, i256, i256, n=8)
        self.conv058 = Conv2dUnit(i256, i256, 1, stride=1)
        self.conv059 = Conv2dUnit(i512, i512, 1, stride=1)

        # ============================= s32 =============================
        self.conv060 = Conv2dUnit(i512, i1024, 3, stride=2, padding=1)
        self.conv061 = Conv2dUnit(i1024, i512, 1, stride=1)
        self.conv062 = Conv2dUnit(i1024, i512, 1, stride=1)
        self.stackResidualBlock05 = StackResidualBlock(i512, i512, i512, n=4)
        self.conv071 = Conv2dUnit(i512, i512, 1, stride=1)
        self.conv072 = Conv2dUnit(i1024, i1024, 1, stride=1)
        # cspdarknet53部分结束

        # fpn部分
        self.conv073 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')
        self.conv074 = Conv2dUnit(i512, i1024, 3, stride=1, padding=1, act='leaky')
        self.conv075 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')
        self.spp = SPP()
        self.conv076 = Conv2dUnit(i512 * 4, i512, 1, stride=1, act='leaky')
        self.conv077 = Conv2dUnit(i512, i1024, 3, stride=1, padding=1, act='leaky')
        self.conv078 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')

        # pan01
        self.conv079 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv080 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.conv081 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.conv082 = Conv2dUnit(i256, i512, 3, stride=1, padding=1, act='leaky')
        self.conv083 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.conv084 = Conv2dUnit(i256, i512, 3, stride=1, padding=1, act='leaky')
        self.conv085 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        # pan01结束

        # pan02
        self.conv086 = Conv2dUnit(i256, i128, 1, stride=1, act='leaky')
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv087 = Conv2dUnit(i256, i128, 1, stride=1, act='leaky')
        self.conv088 = Conv2dUnit(i256, i128, 1, stride=1, act='leaky')
        self.conv089 = Conv2dUnit(i128, i256, 3, stride=1, padding=1, act='leaky')
        self.conv090 = Conv2dUnit(i256, i128, 1, stride=1, act='leaky')
        self.conv091 = Conv2dUnit(i128, i256, 3, stride=1, padding=1, act='leaky')
        self.conv092 = Conv2dUnit(i256, i128, 1, stride=1, act='leaky')
        # pan02结束

        # output_s, 不用concat()
        self.conv093 = Conv2dUnit(i128, i256, 3, stride=1, padding=1, act='leaky')
        self.conv094 = Conv2dUnit(i256, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None)


        # output_m, 需要concat()
        self.conv095 = Conv2dUnit(i128, i256, 3, stride=2, padding=1, act='leaky')

        self.conv096 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.conv097 = Conv2dUnit(i256, i512, 3, stride=1, padding=1, act='leaky')
        self.conv098 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')
        self.conv099 = Conv2dUnit(i256, i512, 3, stride=1, padding=1, act='leaky')
        self.conv100 = Conv2dUnit(i512, i256, 1, stride=1, act='leaky')

        self.conv101 = Conv2dUnit(i256, i512, 3, stride=1, padding=1, act='leaky')
        self.conv102 = Conv2dUnit(i512, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None)

        # output_l, 需要concat()
        self.conv103 = Conv2dUnit(i256, i512, 3, stride=2, padding=1, act='leaky')

        self.conv104 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')
        self.conv105 = Conv2dUnit(i512, i1024, 3, stride=1, padding=1, act='leaky')
        self.conv106 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')
        self.conv107 = Conv2dUnit(i512, i1024, 3, stride=1, padding=1, act='leaky')
        self.conv108 = Conv2dUnit(i1024, i512, 1, stride=1, act='leaky')

        self.conv109 = Conv2dUnit(i512, i1024, 3, stride=1, padding=1, act='leaky')
        self.conv110 = Conv2dUnit(i1024, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None)

    def get_layer(self, name):
        layer = getattr(self, name)
        return layer

    def forward(self, x):
        if torch.cuda.is_available() and self.use_gpu:
            x = x.cuda()

        # cspdarknet53部分
        x = self.conv001(x)

        # ============================= s2 =============================
        x = self.conv002(x)
        s2 = self.conv003(x)
        x = self.conv004(x)
        x = self.stackResidualBlock01(x)
        x = self.conv007(x)
        x = torch.cat([x, s2], dim=1)
        s2 = self.conv008(x)

        # ============================= s4 =============================
        x = self.conv009(s2)
        s4 = self.conv010(x)
        x = self.conv011(x)
        x = self.stackResidualBlock02(x)
        x = self.conv016(x)
        x = torch.cat([x, s4], dim=1)
        s4 = self.conv017(x)

        # ============================= s8 =============================
        x = self.conv018(s4)
        s8 = self.conv019(x)
        x = self.conv020(x)
        x = self.stackResidualBlock03(x)
        x = self.conv037(x)
        x = torch.cat([x, s8], dim=1)
        s8 = self.conv038(x)

        # ============================= s16 =============================
        x = self.conv039(s8)
        s16 = self.conv040(x)
        x = self.conv041(x)
        x = self.stackResidualBlock04(x)
        x = self.conv058(x)
        x = torch.cat([x, s16], dim=1)
        s16 = self.conv059(x)

        # ============================= s32 =============================
        x = self.conv060(s16)
        s32 = self.conv061(x)
        x = self.conv062(x)
        x = self.stackResidualBlock05(x)
        x = self.conv071(x)
        x = torch.cat([x, s32], dim=1)
        s32 = self.conv072(x)
        # cspdarknet53部分结束

        # fpn部分
        x = self.conv073(s32)
        x = self.conv074(x)
        x = self.conv075(x)
        x = self.spp(x)

        x = self.conv076(x)
        x = self.conv077(x)
        fpn_s32 = self.conv078(x)

        # pan01
        x = self.conv079(fpn_s32)
        x = self.upsample1(x)
        s16 = self.conv080(s16)
        x = torch.cat([s16, x], dim=1)
        x = self.conv081(x)
        x = self.conv082(x)
        x = self.conv083(x)
        x = self.conv084(x)
        fpn_s16 = self.conv085(x)
        # pan01结束

        # pan02
        x = self.conv086(fpn_s16)
        x = self.upsample2(x)
        s8 = self.conv087(s8)
        x = torch.cat([s8, x], dim=1)
        x = self.conv088(x)
        x = self.conv089(x)
        x = self.conv090(x)
        x = self.conv091(x)
        x = self.conv092(x)
        # pan02结束

        # output_s, 不用concat()
        output_s = self.conv093(x)
        output_s = self.conv094(output_s)

        # output_m, 需要concat()
        x = self.conv095(x)
        x = torch.cat([x, fpn_s16], dim=1)
        x = self.conv096(x)
        x = self.conv097(x)
        x = self.conv098(x)
        x = self.conv099(x)
        x = self.conv100(x)
        output_m = self.conv101(x)
        output_m = self.conv102(output_m)

        # output_l, 需要concat()
        x = self.conv103(x)
        x = torch.cat([x, fpn_s32], dim=1)
        x = self.conv104(x)
        x = self.conv105(x)
        x = self.conv106(x)
        x = self.conv107(x)
        x = self.conv108(x)
        output_l = self.conv109(x)
        output_l = self.conv110(output_l)

        # 相当于numpy的transpose()，交换下标
        output_l = output_l.permute(0, 2, 3, 1)
        output_m = output_m.permute(0, 2, 3, 1)
        output_s = output_s.permute(0, 2, 3, 1)
        return output_l, output_m, output_s








