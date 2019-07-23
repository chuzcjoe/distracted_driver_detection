# import torch
import torch.nn as nn

layer_config = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
              512, 512, 512],
    'vgg16_reducedfc': [64, 64, 'C', 128, 128, 'C', 256, 256, 256, 'C', 512, 512, 512, 'C',
              512, 512, 512],
    'tiny_vgg': [16, 16, 'C', 32, 32, 'C', 64, 64, 64, 'C', 128, 128, 128, 'C',
              256, 256, 256],
    'tiny_vgg_p3': [16, 16, 32, 32, 64, 64, 64, 'C', 128, 128, 128, 'C',
              256, 256, 256],
}

"""
layer_idx: start from 1, in code need to -1
batch_norm_false: 2 4 5(M) 7 9 10(M) 12 14 16 17(C) 19 21 23 24(M) 26 28 30
batch_norm_true:  3 6 7(M) 10 13 14(M) 17 20 23 24(C) 27 30 33 34(M) 37 40 43
"""


def vgg(cfg, i, batch_norm=False):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def tiny_vgg(cfg, i=3, batch_norm=False):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return layers

def tiny_vgg_p3(cfg, i=3, batch_norm=False):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(256, 256, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

def vgg16():
    return vgg(layer_config['vgg16'], 3)

def tiny_vgg16():
    return tiny_vgg(layer_config['tiny_vgg'], 3)

def tiny_vgg16_p3():
    return tiny_vgg_p3(layer_config['tiny_vgg_p3'], 3)

def vgg16_reducedfc():
    return tiny_vgg(layer_config['vgg16_reducedfc'], 3)
