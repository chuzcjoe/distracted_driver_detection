import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.layers import *


class FSSD(nn.Module):
    # (self, base, extras, head, features, feature_layer, num_classes)
    # FSSD(phase, cfg, base_, extras_, head_, features_, extras[str(size)])
    def __init__(self, phase, cfg, base, extras, head, features, feature_layer):
        super(FSSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.cfg = cfg

        self.priors = None
        self.size = cfg['min_dim']

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)  # is not used

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])

        # Layer learns to scale the l2 normalized features from conv4_3
        # Concat >>> batchnorm
        self.norm = nn.BatchNorm2d(int(feature_layer[0][1][-1] / 2) * len(self.transforms), affine=True)
        self.softmax = nn.Softmax(dim=-1)
        if self.phase == 'test':
            self.detect = DetectOut(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.vgg)):
            x = self.vgg[k](x)
            if k in self.feature_layer:  # [22, 34, 'S']  is get output of relu error
                sources.append(x)  # keep output of layer22 and layer34

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            # TODO: different with lite this one should be change
            if k % 2 == 1:  # conv7_2
                sources.append(x)
        assert len(self.transforms) == len(sources)
        upsize = (sources[0].size()[2], sources[0].size()[3])  # upsample to 38*38

        for k, v in enumerate(self.transforms):  # three transforms layers
            size = None if k == 0 else upsize  # need to upsample
            transformed.append(v(sources[k], size))  # call v.forward(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            if self.priors is None:
                print('Test net init success!')
                return 0
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        elif phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors  # Shape: [2,num_priors*4] ????
            )
        return output


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.up_size = up_size
        # self.up_sample = nn.Upsample(size=(up_size,up_size),mode='bilinear') if up_size != 0 else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode='bilinear')
            # x = self.up_sample(x)
        return x


def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


# [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
def add_extras(base, feature_layer, mbox, num_classes, version='fssd'):
    # Extra layers added to VGG for feature scaling
    extra_layers = []
    feature_transform_layers = []
    pyramid_feature_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    '''
    feature_layer[0]:[[14, 21, 33, 'S'], [256, 512, 1024, 512]],  #concat layer
    '''
    feature_transform_channel = int(feature_layer[0][1][-1] / 2)  # 512/2=256
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
        if 'lite' in version:  # fssd_lite
            if layer == 'S':
                extra_layers += [_conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1)]
                in_channels = depth
            elif layer == '':
                extra_layers += [_conv_dw(in_channels, depth, stride=1, expand_ratio=1)]
                in_channels = depth
            else:
                in_channels = depth
        else:
            if layer == 'S':
                extra_layers += [  # conv7_1 conv7_2
                    nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                    nn.Conv2d(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
                in_channels = depth
            elif layer == '':  # if feature map dimension is 5 or 3
                extra_layers += [
                    nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                    nn.Conv2d(int(depth / 2), depth, kernel_size=3)]
                in_channels = depth
            else:
                in_channels = depth
        feature_transform_layers += [BasicConv(in_channels, feature_transform_channel, kernel_size=1, padding=0)]

    in_channels = len(feature_transform_layers) * feature_transform_channel
    '''
    feature_layer[1]:[['', 'S', 'S', 'S', '', ''], [512, 512, 256, 256, 256, 256]]
    '''
    # in_channels = 3*256
    for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
        if layer == 'S':  #
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=2, padding=1)]
            in_channels = depth
        elif layer == '':  # keep same | dimension is 5 or 3
            pad = (0, 1)[len(pyramid_feature_layers) == 0]
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=1, padding=pad)]
            in_channels = depth
        else:
            AssertionError('Undefined layer')
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (feature_transform_layers, pyramid_feature_layers), (loc_layers, conf_layers)


"""
[0]: add 'S' or '' can add extra layer
[1]: 'S' denote stride = 2

[21, 33, 'S'], [512, 1024, 512]  conv3-conv7
"""
extras = {
    '300': [[[21, 33, 'S'], [512, 1024, 512]],  # concat layer
            [['', 'S', 'S', 'S', '', ''], [512, 512, 256, 256, 256, 256]]],  # pyramid_feature_layers
    '512': [],
}


def build(phase, cfg, base):
    size, num_classes = cfg['min_dim'], cfg['num_classes']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    number_box = [2 * (len(aspect_ratios) + 1) if isinstance(aspect_ratios[0], int)
                  else (len(aspect_ratios) + 1) for aspect_ratios in cfg['aspect_ratios']]

    base_ = base()
    base_, extras_, features_, head_ = add_extras(base_,
                                                  extras[str(size)],
                                                  number_box, num_classes)
    # print ('debug',base_)
    # (self, base, extras, head, features, feature_layer, num_classes)
    return FSSD(phase, cfg, base_, extras_, head_, features_, extras[str(size)])
