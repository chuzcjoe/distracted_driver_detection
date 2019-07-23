import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d\
            (in_planes, in_planes, kernel_size=3, stride=stride, 
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d\
            (in_planes, out_planes, kernel_size=1, 
            stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, 
    # by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 
           512, 512, 512, 512, 512, (1024,2), 1024]


    def __init__(self, num_classes=10,ratio=1.0):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(1, int(16*ratio), kernel_size=3, 
            stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16*ratio))
        self.layers = self._make_layers(in_planes=int(16*ratio),ratio=ratio)
        # self.avgpool = nn.AvgPool2d(5, stride=1)
        # self.fc = nn.Linear(self.cfg[-1], num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal(m.weight)
        #         # nn.init.constant(m.bias, 0)

        #     elif isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal(m.weight)
             


    def _make_layers(self, in_planes,ratio):
        layers = []
        for x in self.cfg_1:
            out_planes = int(x*ratio) if isinstance(x, int) else int(x[0]*ratio)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        # out = self.avgpool(out)
        # # out = F.avg_pool2d(out, 7)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


# def test():
#     net = MobileNet()
#     x = torch.randn(1,3,32,32)
#     y = net(x)
#     print(y.size())
def mobilenet(ratio = 1.0):
    cfg = [16,(24,2), 24, (32,2), 32, 32,(40,2), 
           40]
    layers = []
    in_channels = 3
    conv2d = nn.Conv2d(in_channels, int(16*ratio), kernel_size=3, stride=2, padding=1)
    layers += [conv2d, nn.BatchNorm2d(int(16*ratio)), nn.ReLU(inplace=True)]
    in_planes = int(16*ratio)
    for x in cfg:
        out_planes = int(x*ratio) if isinstance(x, int) else int(x[0]*ratio)
        stride = 1 if isinstance(x, int) else x[1]
        layers.append(Block(in_planes, out_planes, stride))
        in_planes = out_planes

    return layers

# def mobilenet(ratio = 1.0):
#     model = [MobileNet(ratio=ratio)]
#     # model = models.resnet18(pretrained = True)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=modelpath))
#     return model

def build_model(ratio = 1.0):
    return MobileNet(ratio=ratio)
