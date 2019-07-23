import torch
import torch.nn as nn
from lib.layers import *
import os


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class RFB(nn.Module):
    def __init__(self, phase, cfg, base):
        super(RFB, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.phase = phase
        self.priors = None

        self.size = cfg.MODEL.IMAGE_SIZE[0]
        if self.size == 300:
            self.indicator = 3
        elif self.size == 512:
            self.indicator = 5
        else:
            raise Exception("Error: Sorry only SSD300 and SSD512 are supported!")

        self.base = nn.ModuleList(base)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)  # conv4

        extras = self.add_extras(self.size, cfg.MODEL.RFB.EXTRA_CONFIG, 1024)
        self.extras = nn.ModuleList(extras)
        head = self.multibox(self.size, self.base, extras, cfg.MODEL.NUM_PRIOR, self.num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.criterion = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, phase='train', match_result=None, tb_writer=None):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)

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
        return output

    def add_extras(self, size, cfg, i):
        layers = []
        in_channels = i
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    if in_channels == 256 and size == 512:
                        layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=1)]
                    else:
                        layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0, visual=2)]
                else:
                    layers += [BasicRFB(in_channels, v, scale=1.0, visual=2)]
            in_channels = v
        if size == 512:
            layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
            layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
        elif size == 300:
            layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
            layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
            layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
            layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        else:
            raise Exception("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return layers

    def multibox(self, size, vgg, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [-2]
        for k, v in enumerate(vgg_source):
            if k == 0:
                loc_layers += [nn.Conv2d(512,
                                         cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(512,
                                          cfg[k] * num_classes, kernel_size=3, padding=1)]
            else:
                loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                         cfg[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                          cfg[k] * num_classes, kernel_size=3, padding=1)]
        i = 1
        indicator = 0
        if size == 300:
            indicator = 3
        elif size == 512:
            indicator = 5
        else:
            raise Exception("Error: Sorry only RFBNet300 and RFBNet512 are supported!")

        for k, v in enumerate(extra_layers):
            if k < indicator or k % 2 == 0:
                loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                         * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                          * num_classes, kernel_size=3, padding=1)]
                i += 1
        return loc_layers, conf_layers

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


if __name__ == '__main__':
    import os.path as osp
    from lib.utils.config import cfg, merge_cfg_from_file
    from lib.models import model_factory
    cfg_path = osp.join(cfg.GENERAL.CFG_ROOT, 'rfbnet', 'rfb_vgg16_voc_orig.yml')
    merge_cfg_from_file(cfg_path)

    net, priors, layer_dims = model_factory(phase='train', cfg=cfg)
    print(net)
    print(layer_dims)

    # net = build_net('train')
    # print(net)
