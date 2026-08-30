from __future__ import print_function

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .general import GeneralNet

class DenseBottle(nn.Module):
    def __init__(self, in_channels, inner_channel,out_channels):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out=torch.cat([x, self.bottle_neck(x)], 1)
        out = self.relu(out)
        return out

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

class Add(nn.Module):
    def forward(self, x, y):
        return x + y

class WrnBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WrnBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out=torch.add(x if self.equalInOut else self.convShortcut(x), out)
        out=self.relu(out)
        return out


class simfc(nn.Module):
    def __init__(self, in_planes, planes):
        super(simfc, self).__init__()
        self.fc3 = nn.Linear(in_planes, planes)

    def forward(self, x):
        return self.fc3(x)

class Transform(nn.Module):
    def __init__(self, in_planes, planes):
        super(Transform, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.transform(x)

class TransformCon(nn.Module):
    def __init__(self, in_planes, planes):
        super(TransformCon, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )
        self.transform1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x1=self.transform(x)
        x2=self.transform1(x+x1)
        return x2
class TransformLinear(nn.Module):
    def __init__(self, in_planes, planes):
        super(TransformLinear, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes, affine=True),
        )

    def forward(self, x):
        return self.transform(x)
class Transform3(nn.Module):
    def __init__(self, in_planes, planes):
        super(Transform3, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.transform(x)
class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]
class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out
class ShuBasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, is_last=False):
        super(ShuBasicBlock, self).__init__()
        self.is_last = is_last
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        preact = self.bn3(self.conv3(out))
        out = F.relu(preact)
        # out = F.relu(self.bn3(self.conv3(out)))
        preact = torch.cat([x1, preact], 1)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        if self.is_last:
            return out, preact
        else:
            return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.add = Add()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity= self.shortcut(x)
        out = self.add(out, identity)
        preact = out
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out
class BottleneckConv(nn.Module):
    expansion = 8

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BottleneckConv, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.relu=nn.ReLU()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = self.relu(out)

        return out

class ReductionConv1(nn.Module):

    def __init__(self, in_planes, planes):
        super(ReductionConv1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out
class ReductionPool(nn.Module):

    def __init__(self, in_planes, planes):
        super(ReductionPool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.avgpool(x)
        return out
class ReductionConv2(nn.Module):

    def __init__(self, in_planes, planes):
        super(ReductionConv2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out


class ChannelConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=1, stride=1, padding=1, affine=True):
        super(ChannelConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=1,),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SepConv1(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv1, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, channel_input,channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_input, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),

        )

    def forward(self, x):
        return self.op(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.relu = nn.ReLU()
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            out= t + self.conv(x)
            out = self.relu(out)
            return out

        else:
            out=self.conv(x)
            out = self.relu(out)
            return out


class Adaptation_layers(nn.Module):
    def __init__(self):
        super(Adaptation_layers, self).__init__()
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)

    def forward(self, outputs_feature):
        adaptation_feature = []
        adaptation_feature2 = self.linear2(outputs_feature[2])
        adaptation_feature3 = self.linear3(outputs_feature[3])
        adaptation_feature.append(adaptation_feature2)
        adaptation_feature.append(adaptation_feature3)

        return adaptation_feature

class SepAtt(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(SepAtt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            SepConv1(
                channel_in=64,
                channel_out=128
            )
        )

        self.scala2 = nn.Sequential(

            SepConv1(
                channel_in=128,
                channel_out=256
            )
        )
        self.scala3 = nn.Sequential(

            SepConv1(
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            SepConv1(
                channel_in=128,
                channel_out=256
            ),

        )
        self.scala5 = nn.Sequential(

            SepConv1(
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            SepConv1(
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)


        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        # Reduction
        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out24_feature, out14_feature]

class Conv1FcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Conv1FcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            nn.AvgPool2d(32, 32)
        )
        self.scala2 = nn.Sequential(
            nn.AvgPool2d(16, 16)
        )
        self.scala3 = nn.Sequential(
            nn.AvgPool2d(8, 8)
        )
        self.scala4 = nn.Sequential(
            nn.AvgPool2d(4, 4)
        )


        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

class FcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(FcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),
            BasicBlock(
                in_planes=128,
                planes=128,
                stride=1
            ),
        )

        self.scala2 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),
            BasicBlock(
                in_planes=256,
                planes=256,
                stride=1
            ),
        )
        self.scala3 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),
            BasicBlock(
                in_planes=256,
                planes=256,
                stride=1
            ),

        )
        self.scala5 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out12_feature]

class ICLayer1(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ICLayer1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            ReductionConv1(64,512),
            nn.AvgPool2d(32, 32)
        )
        self.scala2 = nn.Sequential(
            ReductionConv1(128, 512),
            nn.AvgPool2d(16, 16)
        )
        self.scala3 = nn.Sequential(
            ReductionConv1(256, 512),
            nn.AvgPool2d(8, 8)
        )
        self.scala4 = nn.Sequential(
            nn.AvgPool2d(4, 4)
        )


        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]
class ICLayer2(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ICLayer2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            ReductionConv2(64,64),
            ReductionConv2(64,64),
            ReductionConv2(64,64),
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            ReductionConv2(128, 128),
            ReductionConv2(128, 128),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            ReductionConv2(256, 256),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.Sequential(
            nn.AvgPool2d(4, 4)
        )


        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(256, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]
class ICLayer3(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ICLayer3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            ReductionConv2(64,64),
            ReductionConv2(64,64),
            ReductionConv2(64,64),
            ReductionConv1(64,512),
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv1(64, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv1(64, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.Sequential(
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv1(64, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala5 = nn.Sequential(
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv2(64, 64),
            ReductionConv1(64, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(
            ReductionConv2(128, 128),
            ReductionConv2(128, 128),
            ReductionConv1(128, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.Sequential(
            ReductionConv2(128, 128),
            ReductionConv2(128, 128),
            ReductionConv1(128, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala8 = nn.Sequential(
            ReductionConv2(128, 128),
            ReductionConv2(128, 128),
            ReductionConv1(128, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala9 = nn.Sequential(
            ReductionConv2(128, 128),
            ReductionConv2(128, 128),
            ReductionConv1(128, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala10 = nn.Sequential(
            ReductionConv2(256, 256),
            ReductionConv1(256, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala11 = nn.Sequential(
            ReductionConv2(256, 256),
            ReductionConv1(256, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala12 = nn.Sequential(
            ReductionConv2(256, 256),
            ReductionConv1(256, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala13 = nn.Sequential(
            ReductionConv2(256, 256),
            ReductionConv1(256, 512),
            nn.AvgPool2d(4, 4)
        )
        self.scala14 = nn.Sequential(
            ReductionConv2(512, 512),
            ReductionConv1(512, 512),
            nn.AvgPool2d(2, 2)
        )
        self.scala15 = nn.Sequential(
            ReductionConv2(512, 512),
            ReductionConv1(512, 512),
            nn.AvgPool2d(2, 2)
        )
        self.scala16 = nn.Sequential(
            ReductionConv2(512, 512),
            ReductionConv1(512, 512),
            nn.AvgPool2d(2, 2)
        )
        self.scala17 = nn.Sequential(
            nn.AvgPool2d(4, 4)
        )


        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        self.fc6 = nn.Linear(512, num_classes)
        self.fc7 = nn.Linear(512, num_classes)
        self.fc8 = nn.Linear(512, num_classes)
        self.fc9 = nn.Linear(512, num_classes)
        self.fc10 = nn.Linear(512, num_classes)
        self.fc11 = nn.Linear(512, num_classes)
        self.fc12 = nn.Linear(512, num_classes)
        self.fc13 = nn.Linear(512, num_classes)
        self.fc14 = nn.Linear(512, num_classes)
        self.fc15 = nn.Linear(512, num_classes)
        self.fc16 = nn.Linear(512, num_classes)
        self.fc17 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)
        out5_feature = self.scala5(feature_list[4]).view(x[0].size(0), -1)
        out6_feature = self.scala6(feature_list[5]).view(x[0].size(0), -1)
        out7_feature = self.scala7(feature_list[6]).view(x[0].size(0), -1)
        out8_feature = self.scala8(feature_list[7]).view(x[0].size(0), -1)
        out9_feature = self.scala9(feature_list[8]).view(x[0].size(0), -1)
        out10_feature = self.scala10(feature_list[9]).view(x[0].size(0), -1)
        out11_feature = self.scala11(feature_list[10]).view(x[0].size(0), -1)
        out12_feature = self.scala12(feature_list[11]).view(x[0].size(0), -1)
        out13_feature = self.scala13(feature_list[12]).view(x[0].size(0), -1)
        out14_feature = self.scala14(feature_list[13]).view(x[0].size(0), -1)
        out15_feature = self.scala15(feature_list[14]).view(x[0].size(0), -1)
        out16_feature = self.scala16(feature_list[15]).view(x[0].size(0), -1)
        out17_feature = self.scala17(feature_list[16]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)
        out5 = self.fc5(out5_feature)
        out6 = self.fc6(out6_feature)
        out7 = self.fc7(out7_feature)
        out8 = self.fc8(out8_feature)
        out9 = self.fc9(out9_feature)
        out10 = self.fc10(out10_feature)
        out11 = self.fc11(out11_feature)
        out12 = self.fc12(out12_feature)
        out13 = self.fc13(out13_feature)
        out14 = self.fc14(out14_feature)
        out15 = self.fc15(out15_feature)
        out16 = self.fc16(out16_feature)
        out17 = self.fc17(out17_feature)

        return [out17,out16, out15, out14, out13,out12, out11, out10, out9,out8, out7, out6, out5,out4, out3, out2, out1], \
               [out17_feature, out16_feature, out15_feature, out14_feature, out13_feature,out12_feature, out11_feature, out10_feature,
                out9_feature,out8_feature, out7_feature, out6_feature, out5_feature,out4_feature, out3_feature, out2_feature, out1_feature]
class ICLayer4(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ICLayer4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            BasicBlock(64,128,2),
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            BasicBlock(64, 128,2),
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            BasicBlock(64, 128,2),
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.Sequential(
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala5 = nn.Sequential(
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.Sequential(
            BasicBlock(128, 256,2),
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala8 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala9 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala10 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala11 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala12 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala13 = nn.Sequential(
            BasicBlock(256, 512,2),
            nn.AvgPool2d(4, 4)
        )
        self.scala14 = nn.Sequential(
            BasicBlock(512, 512,2),
            nn.AvgPool2d(2, 2)
        )
        self.scala15 = nn.Sequential(
            BasicBlock(512, 512,2),
            nn.AvgPool2d(2, 2)
        )
        self.scala16 = nn.Sequential(
            nn.AvgPool2d(4, 4)
        )



        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        self.fc6 = nn.Linear(512, num_classes)
        self.fc7 = nn.Linear(512, num_classes)
        self.fc8 = nn.Linear(512, num_classes)
        self.fc9 = nn.Linear(512, num_classes)
        self.fc10 = nn.Linear(512, num_classes)
        self.fc11 = nn.Linear(512, num_classes)
        self.fc12 = nn.Linear(512, num_classes)
        self.fc13 = nn.Linear(512, num_classes)
        self.fc14 = nn.Linear(512, num_classes)
        self.fc15 = nn.Linear(512, num_classes)
        self.fc16 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)
        out5_feature = self.scala5(feature_list[4]).view(x[0].size(0), -1)
        out6_feature = self.scala6(feature_list[5]).view(x[0].size(0), -1)
        out7_feature = self.scala7(feature_list[6]).view(x[0].size(0), -1)
        out8_feature = self.scala8(feature_list[7]).view(x[0].size(0), -1)
        out9_feature = self.scala9(feature_list[8]).view(x[0].size(0), -1)
        out10_feature = self.scala10(feature_list[9]).view(x[0].size(0), -1)
        out11_feature = self.scala11(feature_list[10]).view(x[0].size(0), -1)
        out12_feature = self.scala12(feature_list[11]).view(x[0].size(0), -1)
        out13_feature = self.scala13(feature_list[12]).view(x[0].size(0), -1)
        out14_feature = self.scala14(feature_list[13]).view(x[0].size(0), -1)
        out15_feature = self.scala15(feature_list[14]).view(x[0].size(0), -1)
        out16_feature = self.scala16(feature_list[15]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)
        out5 = self.fc5(out5_feature)
        out6 = self.fc6(out6_feature)
        out7 = self.fc7(out7_feature)
        out8 = self.fc8(out8_feature)
        out9 = self.fc9(out9_feature)
        out10 = self.fc10(out10_feature)
        out11 = self.fc11(out11_feature)
        out12 = self.fc12(out12_feature)
        out13 = self.fc13(out13_feature)
        out14 = self.fc14(out14_feature)
        out15 = self.fc15(out15_feature)
        out16 = self.fc16(out16_feature)

        return [out16, out15, out14, out13,out12, out11, out10, out9,out8, out7, out6, out5,out4, out3, out2, out1], \
               [out16_feature, out15_feature, out14_feature, out13_feature,out12_feature, out11_feature, out10_feature,
                out9_feature,out8_feature, out7_feature, out6_feature, out5_feature,out4_feature, out3_feature, out2_feature, out1_feature]

class SepFcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(SepFcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            SepConv(
                channel_input=64,
                channel_in=64,
                channel_out=128
            ),
        )

        self.scala2 = nn.Sequential(

            SepConv(
                channel_input=128,
                channel_in=128,
                channel_out=256
            ),
        )
        self.scala3 = nn.Sequential(

            SepConv(
                channel_input=256,
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            SepConv(
                channel_input=128,
                channel_in=128,
                channel_out=256
            ),

        )
        self.scala5 = nn.Sequential(

            SepConv(
                channel_input=256,
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            SepConv(
                channel_input=256,
                channel_in=256,
                channel_out=512
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out12_feature]
class WrnFcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(WrnFcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            WrnBlock(
                in_planes=32,
                out_planes=64,
                stride=2,
                dropRate=0.0
            )
        )

        self.scala2 = nn.Sequential(

            WrnBlock(
                in_planes=64,
                out_planes=128,
                stride=2,
                dropRate=0.0
            )
        )
        self.scala3 = nn.Sequential(

            WrnBlock(
                in_planes=128,
                out_planes=256,
                stride=2,
                dropRate=0.0
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            WrnBlock(
                in_planes=64,
                out_planes=128,
                stride=2,
                dropRate=0.0
            )

        )
        self.scala5 = nn.Sequential(

            WrnBlock(
                in_planes=128,
                out_planes=256,
                stride=2,
                dropRate=0.0
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            WrnBlock(
                in_planes=128,
                out_planes=256,
                stride=2,
                dropRate=0.0
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(256, num_classes)
        self.fc3 = nn.Linear(256, num_classes)
        self.fc4 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out12_feature]
class ShuFcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ShuFcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            DownBlock(116,232,2),
            ShuBasicBlock(232)
        )

        self.scala2 = nn.Sequential(

            DownBlock(232, 464, 2),
            ShuBasicBlock(464)
        )
        self.scala3 = nn.Sequential(

            DownBlock(464, 1024, 2),
            ShuBasicBlock(1024),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            DownBlock(232, 464, 2),
            ShuBasicBlock(464)

        )
        self.scala5 = nn.Sequential(

            DownBlock(464, 1024, 2),
            ShuBasicBlock(1024),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            DownBlock(464, 1024, 2),
            ShuBasicBlock(1024),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc3 = nn.Linear(1024, num_classes)
        self.fc4 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out12_feature]

class MobileFcLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(MobileFcLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            InvertedResidual(
                inp=12,
                oup=24,
                stride=2,
                expand_ratio=6,
            )
        )

        self.scala2 = nn.Sequential(

            InvertedResidual(
                inp=24,
                oup=48,
                stride=2,
                expand_ratio=6,
            )
        )
        self.scala3 = nn.Sequential(

            InvertedResidual(
                inp=48,
                oup=160,
                stride=2,
                expand_ratio=6,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            InvertedResidual(
                inp=24,
                oup=48,
                stride=2,
                expand_ratio=6,
            )

        )
        self.scala5 = nn.Sequential(

            InvertedResidual(
                inp=48,
                oup=160,
                stride=2,
                expand_ratio=6,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            InvertedResidual(
                inp=48,
                oup=160,
                stride=2,
                expand_ratio=6,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(160, num_classes)
        self.fc2 = nn.Linear(160, num_classes)
        self.fc3 = nn.Linear(160, num_classes)
        self.fc4 = nn.Linear(160, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out12_feature]


class ReLayer1(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ReLayer1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),

        )

        self.scala2 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala3 = nn.Sequential(


            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2,
            ),
            nn.AvgPool2d(4, 4)

        )

        self.scala4 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala5 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            nn.AvgPool2d(4, 4)


        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            nn.AvgPool2d(4, 4)

        )
        self.attention1 = nn.Sequential(
            Transform(in_planes=512,planes=512),
            nn.Sigmoid()
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out34_feature = self.scala6(feature_list[2])
        out3_feature = out34_feature.view(x[0].size(0), -1)

        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [feature_list[3], out34_feature, out23_feature, out13_feature]

class IntermediaterR1(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(IntermediaterR1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),

        )

        self.scala2 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala3 = nn.Sequential(


            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2,
            ),
            nn.AvgPool2d(4, 4)

        )

        self.scala4 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala5 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            nn.AvgPool2d(4, 4)


        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            nn.AvgPool2d(4, 4)


        )
        self.attention1 = nn.Sequential(
            Transform(in_planes=512,planes=512),
            nn.Sigmoid()
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out34_feature = self.scala6(feature_list[2])
        out3_feature = out34_feature.view(x[0].size(0), -1)

        out44_feature = self.scala7(feature_list[3])
        out4_feature=out44_feature.view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out13_feature,out23_feature, out34_feature,out44_feature]

class IncrementLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(IncrementLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.incre1 = nn.Sequential(
            Transform(in_planes=512, planes=512),
        )
        self.incre2 = nn.Sequential(
            Transform(in_planes=512, planes=512),
        )
        self.incre3 = nn.Sequential(
            Transform(in_planes=512, planes=512),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x
        out0_feature =feature_list[0].unsqueeze(-1).unsqueeze(-1)
        out1_feature = self.incre1(feature_list[1].unsqueeze(-1).unsqueeze(-1))
        out2_feature = self.incre2(feature_list[2].unsqueeze(-1).unsqueeze(-1))
        out3_feature = self.incre3(feature_list[3].unsqueeze(-1).unsqueeze(-1))

        # out0_feature=out0_feature.view(x[0].size(0), -1)
        # out1_feature = out1_feature.view(x[0].size(0), -1)
        # out2_feature = out2_feature.view(x[0].size(0), -1)
        # out3_feature = out3_feature.view(x[0].size(0), -1)

        return  [out0_feature,out1_feature,out2_feature,out3_feature]

class Incrementfc(nn.Module):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Incrementfc, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x1,x2):
        feature_list1 = x1
        feature_list2 = x2

        feature1=feature_list1[3]
        feature1 = feature1.view(x1[0].size(0), -1)

        feature1 = feature_list1[3]
        feature1 = feature1.view(x1[0].size(0), -1)

        feature2 = feature_list1[3] + feature_list1[2] + 0.5 * (
                    feature_list1[2] - feature_list2[3].view(x2[0].size(0), -1))
        feature2 = feature2.view(x1[0].size(0), -1)

        feature3 = feature_list1[2] + feature_list1[1] + 0.5 * (
                    feature_list1[1] - feature_list2[2].view(x2[0].size(0), -1))
        feature3 = feature3.view(x1[0].size(0), -1)

        feature4 = feature_list1[1] + feature_list1[0] + 0.5 * (
                    feature_list1[0] - feature_list2[1].view(x2[0].size(0), -1))
        feature4 = feature4.view(x1[0].size(0), -1)

        out1 = self.fc1(feature1)
        out2 = self.fc2(feature2)
        out3 = self.fc3(feature3)
        out4 = self.fc4(feature4)

        return  [out4, out3, out2, out1]

class DenseLayer(GeneralNet):
    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(DenseLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            Transition(
                in_channels=256,
                out_channels=256),
            DenseBottle(
                in_channels=256,
                inner_channel=128,
                out_channels=256,
            ),

        )

        self.scala2 = nn.Sequential(
            Transition(
                in_channels=512,
                out_channels=512),
            DenseBottle(
                in_channels=512,
                inner_channel=128,
                out_channels=512,
            ),



        )
        self.scala3 = nn.Sequential(
            Transition(
                in_channels=1024,
                out_channels=512),
            DenseBottle(
                in_channels=512,
                inner_channel=128,
                out_channels=512,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            Transition(
                in_channels=512,
                out_channels=512),
            DenseBottle(
                in_channels=512,
                inner_channel=128,
                out_channels=512,
            ),


        )
        self.scala5 = nn.Sequential(

            Transition(
                in_channels=1024,
                out_channels=512),
            DenseBottle(
                in_channels=512,
                inner_channel=128,
                out_channels=512,
            ),
            nn.AvgPool2d(4, 4)


        )
        self.scala6 = nn.Sequential(
            Transition(
                in_channels=1024,
                out_channels=512),
            DenseBottle(
                in_channels=512,
                inner_channel=128,
                out_channels=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc3 = nn.Linear(1024, num_classes)
        self.fc4 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, feature_list[2], feature_list[1], out1_feature]
class BottleLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(BottleLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            Bottleneck(
                in_planes=256,
                planes=128,
                stride=2
            ),

        )

        self.scala2 = nn.Sequential(

            Bottleneck(
                in_planes=512,
                planes=256,
                stride=2
            ),


        )
        self.scala3 = nn.Sequential(

            Bottleneck(
                in_planes=1024,
                planes=512,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            Bottleneck(
                in_planes=512,
                planes=256,
                stride=2
            ),


        )
        self.scala5 = nn.Sequential(

            Bottleneck(
                in_planes=1024,
                planes=512,
                stride=2
            ),


            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            Bottleneck(
                in_planes=1024,
                planes=512,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(2048, num_classes)
        self.fc3 = nn.Linear(2048, num_classes)
        self.fc4 = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, feature_list[2], feature_list[1], out1_feature]
class DifferentReLayer(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(DifferentReLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            SepConv(
                channel_input=64,
                channel_in=64,
                channel_out=128
            ),

        )

        self.scala2 = nn.Sequential(

            SepConv(
                channel_input=128,
                channel_in=128,
                channel_out=256
            ),

        )
        self.scala3 = nn.Sequential(

            SepConv(
                channel_input=256,
                channel_in=256,
                channel_out=512
            ),

            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            Bottleneck(
                in_planes=128,
                planes=64,
                stride=2
            ),

        )
        self.scala5 = nn.Sequential(

            Bottleneck(
                in_planes=256,
                planes=128,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2,
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])

        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

class ReLayer8(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ReLayer8, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),

        )

        self.scala2 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala3 = nn.Sequential(


            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2,
            ),


            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),

        )

        self.scala5 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2,
            ),

            nn.AvgPool2d(4, 4)
        )

        self.scala7 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala8 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),


            nn.AvgPool2d(4, 4)
        )
        self.scala9 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),

        )
        self.scala10 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala11 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala12 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala13 = nn.AvgPool2d(4, 4)
        self.scala14 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        self.fc6 = nn.Linear(512, num_classes)
        self.fc7 = nn.Linear(512, num_classes)
        self.fc8 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out22_feature = self.scala5(out2_feature)
        out23_feature = self.scala6(out22_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala7(feature_list[2])
        out33_feature = self.scala8(out3_feature)
        out34_feature = out33_feature.view(x[0].size(0), -1)

        out4_feature = self.scala9(feature_list[3])
        out43_feature = self.scala10(out4_feature)
        out44_feature = out43_feature.view(x[0].size(0), -1)

        out5_feature = self.scala11(feature_list[4]).view(x[0].size(0), -1)
        out6_feature = self.scala12(feature_list[5]).view(x[0].size(0), -1)

        out7_feature = self.scala13(feature_list[6]).view(x[0].size(0), -1)
        out8_feature = self.scala14(feature_list[7]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out24_feature)
        out3 = self.fc3(out34_feature)
        out4 = self.fc4(out44_feature)
        out5 = self.fc5(out5_feature)
        out6 = self.fc6(out6_feature)
        out7 = self.fc7(out7_feature)
        out8 = self.fc8(out8_feature)

        return [out8, out7, out6, out5,out4, out3, out2, out1], [out8_feature, out7_feature, out6_feature, out5_feature,out4_feature, out34_feature, out2_feature, out14_feature]

class ReLayer2(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(ReLayer2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(

            BasicBlock(
                in_planes=64,
                planes=128,
                stride=2
            ),
            BasicBlock(
                in_planes=128,
                planes=128,
                stride=1
            ),

        )

        self.scala2 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),
            BasicBlock(
                in_planes=256,
                planes=256,
                stride=1
            ),

        )
        self.scala3 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),

            nn.AvgPool2d(4, 4)
        )

        self.scala4 = nn.Sequential(

            BasicBlock(
                in_planes=128,
                planes=256,
                stride=2
            ),
            BasicBlock(
                in_planes=256,
                planes=256,
                stride=1
            ),


        )
        self.scala5 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),


            nn.AvgPool2d(4, 4)
        )
        self.scala6 = nn.Sequential(

            BasicBlock(
                in_planes=256,
                planes=512,
                stride=2
            ),
            BasicBlock(
                in_planes=512,
                planes=512,
                stride=1
            ),

            nn.AvgPool2d(4, 4)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0])
        out12_feature = self.scala2(out1_feature)
        out13_feature = self.scala3(out12_feature)
        out14_feature = out13_feature.view(x[0].size(0), -1)

        out2_feature = self.scala4(feature_list[1])
        out23_feature = self.scala5(out2_feature)
        out24_feature = out23_feature.view(x[0].size(0), -1)

        out3_feature = self.scala6(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala7(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out14_feature)
        out2 = self.fc2(out14_feature+out24_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class FcLayer2(nn.Module):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(FcLayer2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            SepConv1(
                channel_in=64,
                channel_out=512,
                stride=8,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv1(
                channel_in=128,
                channel_out=512,
                stride=4,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            SepConv1(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class FcLayer3(nn.Module):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(FcLayer3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala1 = nn.Sequential(
            ChannelConv(
                channel_in=64,
                channel_out=512,
            ),
            nn.AvgPool2d(32, 32)
        )
        self.scala2 = nn.Sequential(
            ChannelConv(
                channel_in=64,
                channel_out=512,
            ),
            nn.AvgPool2d(32, 32)
        )

        self.scala3 = nn.Sequential(
            ChannelConv(
                channel_in=128,
                channel_out=512,
            ),
            nn.AvgPool2d(16, 16)
        )
        self.scala4 = nn.Sequential(
            ChannelConv(
                channel_in=128,
                channel_out=512,
            ),
            nn.AvgPool2d(16, 16)
        )
        self.scala5 = nn.Sequential(
            ChannelConv(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(8, 8)
        )
        self.scala6 = nn.Sequential(
            ChannelConv(
                channel_in=256,
                channel_out=512,
            ),
            nn.AvgPool2d(8, 8)
        )
        self.scala7 = nn.AvgPool2d(4, 4)
        self.scala8 = nn.AvgPool2d(4, 4)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)
        self.fc5 = nn.Linear(512, num_classes)
        self.fc6 = nn.Linear(512, num_classes)
        self.fc7 = nn.Linear(512, num_classes)
        self.fc8 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x

        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x[0].size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x[0].size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x[0].size(0), -1)
        out5_feature = self.scala5(feature_list[4]).view(x[0].size(0), -1)
        out6_feature = self.scala6(feature_list[5]).view(x[0].size(0), -1)
        out7_feature = self.scala7(feature_list[6]).view(x[0].size(0), -1)
        out8_feature = self.scala8(feature_list[7]).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)
        out5 = self.fc5(out5_feature)
        out6 = self.fc6(out6_feature)
        out7 = self.fc7(out7_feature)
        out8 = self.fc8(out8_feature)

        return [out8, out7, out6, out5, out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature,
                                                                  out1_feature]
class Scala1(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Scala1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala3 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 256, 2),
            BasicBlock(256, 512, 2),
            nn.AvgPool2d(4, 4)
        )
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        feature_list = x
        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out1 = self.fc1(out1_feature)

        return [out1],[out1_feature]
class Scala2(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Scala2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala4 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 512, 2),
            nn.AvgPool2d(4, 4)
        )
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        feature_list = x
        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out1 = self.fc1(out1_feature)

        return [out1],[out1_feature]
class Scala3(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Scala3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala13 = nn.Sequential(
            BasicBlock(256, 512, 2),
            nn.AvgPool2d(4, 4)
        )
        self.fc13 = nn.Linear(512, num_classes)

    def forward(self, x):
        feature_list = x
        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out1 = self.fc1(out1_feature)

        return [out1],[out1_feature]
class Scala4(GeneralNet):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(Scala4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scala15 = nn.Sequential(
            BasicBlock(512, 512, 2),
            nn.AvgPool2d(2, 2)
        )
        self.fc15 = nn.Linear(512, num_classes)

    def forward(self, x):
        feature_list = x
        out1_feature = self.scala1(feature_list[0]).view(x[0].size(0), -1)
        out1 = self.fc1(out1_feature)

        return [out1],[out1_feature]

class FcLayer1(nn.Module):

    def __init__(self, num_classes=100, zero_init_residual=False,
                 norm_layer=None):
        super(FcLayer1, self).__init__()

        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(256, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        feature_list = x
        out1_feature = F.avg_pool2d(feature_list[0], 32).view(x[0].size(0), -1)
        out2_feature = F.avg_pool2d(feature_list[1], 16).view(x[0].size(0), -1)
        out3_feature = F.avg_pool2d(feature_list[2], 8).view(x[0].size(0), -1)
        out4_feature = F.avg_pool2d(feature_list[3], 4).view(x[0].size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


class LinearTester(nn.Module):

    def __init__(self, input_size, output_size, layers=3, init_weights=True,
                 fix_p=False, affine=False, bn=True, kernel_size=3, padding=1, instance_bn=True, state=0):
        super(LinearTester, self).__init__()
        self.layers = layers
        # size = [C,H,W]
        self.affine = affine
        self.input_size = input_size
        self.output_size = output_size
        self.bn = bn
        self.state = state
        self.nonLinearLayers_p_pre = nn.Parameter(torch.tensor([1.0, 1.0]).cuda(), requires_grad=(not fix_p))
        self.nonLinearLayers_p = self.get_p()
        self.instance_bn = instance_bn
        # self._make_nonLinearLayers()
        # self._make_linearLayers()
        if init_weights:
            self._initialize_weights()
        # For record
        self.nonLinearLayersRecord = torch.zeros((layers, *self.output_size)).cuda()
        # def _make_linearLayers(self):
        inCh = self.input_size[0]
        outCh = self.output_size[0]
        if self.bn:
            if not self.instance_bn:
                self.linearLayers_bn = nn.BatchNorm2d(inCh, affine=self.affine, track_running_stats=False)
            else:
                self.linearLayers_bn = nn.InstanceNorm2d(inCh, affine=self.affine, track_running_stats=False)
        linearLayers_conv = []
        nonLinearLayers_ReLU = []
        for x in range(self.layers):
            linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=kernel_size, padding=padding, bias=False)]
            nonLinearLayers_ReLU += [nn.ReLU(inplace=True)]
        self.linearLayers_conv = nn.ModuleList(linearLayers_conv)
        self.nonLinearLayers_ReLU = nn.ModuleList(nonLinearLayers_ReLU)

        if not instance_bn:
            self.nonLinearLayers_norm = nn.Parameter(torch.ones(self.layers, self.output_size[0]),
                                                     requires_grad=False)
            self.running_times = nn.Parameter(torch.zeros(self.layers, dtype=torch.long), requires_grad=False)

        else:
            self.nonLinearLayers_norm = torch.ones(self.layers - 1, 1, self.output_size[0]).cuda()
            # self.nonLinearLayers_norm = torch.ones(self.layers - 1).cuda(self.gpu_id)

    def get_p(self):
        return nn.Sigmoid()(self.nonLinearLayers_p_pre)

    def forward(self, x):
        self.nonLinearLayers_p = self.get_p()
        if self.bn:
            x = self.linearLayers_bn(x)
        else:
            x = self.my_bn(self.layers - 1, x)

        out = self.linear(self.state, x, torch.zeros_like(x))
        for i in range(1 + self.state, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        return out

    def my_bn(self, i, out, momentum=0.1, eps=1e-5, rec=False, yn=False):
        if not self.instance_bn:
            if self.training:
                a = out.transpose(0, 1).reshape([out.shape[1], -1]).var(-1).sqrt() + eps
                if self.running_times[i] == 0:
                    self.nonLinearLayers_norm[i] = a
                else:
                    self.nonLinearLayers_norm[i] = (1 - momentum) * self.nonLinearLayers_norm[i] + momentum * a
                self.running_times[i] += 1
                a_ = a.reshape(1, out.shape[1], 1, 1)
            else:
                a_ = self.nonLinearLayers_norm[i].reshape(1, out.shape[1], 1, 1)

            a_ = a_.repeat(out.shape[0], 1, out.shape[2], out.shape[3])
            out = out / a_
            return out
        else:
            if not yn:
                a = out.data.reshape([*out.shape[:-2], self.output_size[1] * self.output_size[2]])
                # a = out.data.reshape([*out.shape[:-3],-1]).var(-1).sqrt() \
                #     + eps
                if a.size()[-1] == 1:
                    a = torch.ones_like(a)
                    if rec:
                        self.nonLinearLayers_norm[i] = a.reshape([*a.shape[:-1]])
                else:
                    a = a.var(-1).sqrt() + eps
                    if rec:
                        self.nonLinearLayers_norm[i] = a.squeeze(0)
            else:
                a = self.nonLinearLayers_norm[i]
            a = a.reshape([*out.shape[:-2], 1, 1])
            # a = a.reshape([out.shape[0],1, 1, 1])
            out = out / a
            return out

    def nonLinear(self, i, out, rec=False):
        out = self.my_bn(i, out, rec=rec)
        out = self.nonLinearLayers_ReLU[i](out)
        if rec:
            self.nonLinearLayersRecord[i] = torch.gt(out, 0)  # .reshape(self.input_size)
        out = self.nonLinearLayers_p[i] * out
        return out

    def linear(self, i, x, out):
        out = x + out
        out = self.linearLayers_conv[i](out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""

    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""

    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""

    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu

        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv1 = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
            self.conv2 = nn.Conv2d(t_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
            self.conv3 = nn.Conv2d(t_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn1 = nn.BatchNorm2d(t_C)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(t_C)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(t_C)
        self.relu3 = nn.ReLU(inplace=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_relu:
            x = self.relu1(self.bn1(x))
        else:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.use_relu:
            x = self.relu2(self.bn2(x))
        else:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.use_relu:
            x = self.relu3(self.bn3(x))
        else:
            x = self.bn3(x)
        return x


class Regress(nn.Module):
    """Simple Linear Regression for hints"""

    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""

    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels * pool_size * pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
