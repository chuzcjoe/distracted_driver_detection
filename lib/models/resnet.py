# import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class ResidualBlock_small(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, conv_2=False):
        super(ResidualBlock_small, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_2 = conv_2
        if conv_2 == True:
            self.conv1_1 = conv1x1(in_channels, out_channels, stride=stride)
            self.bn1_2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv_2:
            residual = self.conv1_1(x)
            residual = self.bn1_2(residual)
        out += residual
        out = self.relu(out)
        return out


# def res10():

#     layers = []
#     in_channels = 3
#     all_channels = 12
#     conv2d = nn.Conv2d(in_channels, all_channels, kernel_size=3, stride=2, padding=1)
#     layers += [conv2d, nn.BatchNorm2d(all_channels), nn.ReLU(inplace=True)]
#     layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0,ceil_mode=True)]
#     layers += [ResidualBlock_small(all_channels,all_channels,1,True)]
#     layers += [ResidualBlock_small(all_channels,all_channels,2,True)]
#     layers += [ResidualBlock_small(all_channels,all_channels,2,True)]
#     return layers
def res10():

    layers = []
    in_channels = 3
    all_channels = 12
    conv2d = nn.Conv2d(in_channels, 19, kernel_size=3, stride=2, padding=1)
    layers += [conv2d, nn.BatchNorm2d(19), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0,ceil_mode=True)]
    layers += [ResidualBlock_small(19,32,1,True)]
    layers += [ResidualBlock_small(32,64,2,True)]
    layers += [ResidualBlock_small(64,96,2,True)]
    return layers

# def res10_t():

#     layers = []
#     in_channels = 3
#     conv2d = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
#     layers += [conv2d, nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
#     layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0,ceil_mode=True)]
#     layers += [ResidualBlock_small(16,32,1,True)]
#     layers += [ResidualBlock_small(32,64,2,True)]
#     layers += [ResidualBlock_small(64,96,2,True)]
#     return layers


def res10_t():

    layers = []
    in_channels = 3
    conv2d = nn.Conv2d(in_channels, 19, kernel_size=3, stride=2, padding=1)
    layers += [conv2d, nn.BatchNorm2d(19), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)]
    layers += [ResidualBlock_small(19, 32, 1, True)]
    layers += [ResidualBlock_small(32, 64, 2, True)]
    layers += [ResidualBlock_small(64, 96, 2, True)]
    return layers