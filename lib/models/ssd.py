import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from lib.layers import *

block_extras = ['vgg16', 'tiny_vgg16']
class SSD(nn.Module):
    """Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """
    #SSD(phase, base_, extras_, head_, feature_layer, num_classes)
    def __init__(self, phase, cfg, base):
        super(SSD, self).__init__()
        if phase != "train" and phase != "eval":
            raise Exception("ERROR: Input phase: {} not recognized".format(phase))
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.priors = None
        self.feature_layer = cfg.MODEL.SSD.EXTRA_CONFIG
        # SSD network
        self.base = nn.ModuleList(base)
        # self.norm = L2Norm(self.feature_layer[1][0], 20)
        extras, head = add_extras(self.feature_layer, cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES, lite=cfg.MODEL.LITE)
        if extras is not []:
            self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = None

    def forward(self, x, phase='train', match_result=None, tb_writer=None):

        sources, loc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if (k in self.feature_layer[0]) or ((k - len(self.base)) in self.feature_layer[0]): #e.g. vgg: conv4_3_relu fc7_relu
                # if len(sources) == 0 and (self.cfg.MODEL.BASE in block_extras):
                #     # s = self.norm(x)
                #     sources.append(s)
                # else:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            # lite is different in here, should be changed, not need relu
            x = v(x)
            if self.cfg.MODEL.BASE in block_extras: #e.g. base is vgg16
                x = F.relu(x, inplace=True)  
                if k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)
        
        #print('extras', torch.mean(x))
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if phase == 'eval':
            output = loc, self.softmax(conf)
        else:
            output = self.criterion((loc, conf), match_result, tb_writer) \
                if self.criterion is not None else None
            # output = loc, conf
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    #m == model.conv1
                if self.cfg.MODEL.INIT_WEIGHTS == 'xavier':
                    init.xavier_uniform(m.weight.data)
                elif self.cfg.MODEL.INIT_WEIGHTS == 'msra':
                    init.kaiming_normal(m.weight.data, mode='fan_in')   #==msra
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

#[22, 34, 'S', 'S', 'S', ''], [512, 1024, 512, 256, 256, 256]
def add_extras(feature_layer, num_box, num_classes, lite=False):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], num_box):
        if lite:
            if layer == 'S':
                extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
                in_channels = depth
            elif layer == '':
                extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
                in_channels = depth
            else:
                in_channels = depth
        else:
            if layer == 'S':
                extra_layers += [
                        nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                        nn.Conv2d(int(depth/2), depth, kernel_size=3, stride=2, padding=1)  ]
                in_channels = depth
            elif layer == '':
                extra_layers += [
                        nn.Conv2d(in_channels, int(depth/2), kernel_size=1),
                        nn.Conv2d(int(depth/2), depth, kernel_size=3)  ]
                in_channels = depth
            else:
                in_channels = depth
        
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return extra_layers, (loc_layers, conf_layers)

class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride=1, padding=0, expand_ratio=1):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup
    
    def forward(self, x):
        return self.conv(x)

#TODO use function maybe in future
class _conv_dw_nouse(nn.Module):
    def __init__(self, inp, oup, stride=1, padding=0, expand_ratio=1):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup * expand_ratio),
            nn.ReLU6(inplace=True), #min(max(x, 0), 6)
            # dw
            nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
            nn.BatchNorm2d(oup * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
    
    def forward(self, x):
        return self.conv(x)

def build_ssd(phase, cfg, base, num_box=None):
    num_classes = cfg['num_classes']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    
    print('number_box', num_box)
    
    feature_layer = cfg['feature_layer']
    base_, extras_, head_ = add_extras(base(), feature_layer,
                                     num_box, num_classes)
    return SSD(phase, base_, extras_, head_, feature_layer, cfg)

def build_ssd_lite(phase, cfg, base, num_box=None):
    num_classes = cfg['num_classes']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    
    print('number_box', num_box)
    
    feature_layer = cfg['feature_layer']
    base_, extras_, head_ = add_extras(base(), feature_layer,
                                     num_box, num_classes, lite=True)
    return SSD(phase, base_, extras_, head_, feature_layer, cfg)
