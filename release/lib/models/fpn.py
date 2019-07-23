import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *


class FPN(nn.Module):
    def __init__(self, phase, cfg, base):
        super(FPN, self).__init__()
        if phase != "test" and phase != "train" and phase != "eval":
            raise Exception("ERROR: Input phase: {} not recognized".format(phase))

        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.priors = None
        self.image_size = cfg.MODEL.IMAGE_SIZE

        # SSD network
        self.base = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)  # TODO automate this

        extras, features_ = add_extras(cfg.MODEL.FPN.EXTRA_CONFIG)
        head = multibox(features_[1], cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.transforms = nn.ModuleList(features_[0])  # lateral layers 1x1
        self.pyramids = nn.ModuleList(features_[1])  # final output layers 3x3

        self.softmax = nn.Softmax(dim=-1)
        if self.phase == 'test':  # TODO add to config
            self.detect = DetectOut(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]

        # apply bases layers and cache source layer outputs
        for idx in range(len(self.base)):
            x = self.base[idx](x)
            if idx == 22:
                s = self.L2Norm(x)
                sources.append(s)
        sources.append(x)  # keep output of layer22 and layer34

        # apply extra layers and cache source layer outputs
        for idx, func in enumerate(self.extras):
            x = F.relu(func(x), inplace=True)
            if idx % 2 == 1:  #
                sources.append(x)
        assert len(self.transforms) == len(sources)

        for idx, func in enumerate(self.transforms):  # transforms layers
            transformed.append(func(sources[idx]))

        for idx, func in enumerate(self.pyramids):  # pyramids layers
            # TODO upsize is feat_dim
            upsize = (transformed[-1 - idx].size()[2], transformed[-1 - idx].size()[3])
            size = None if idx == 0 else upsize  # need to upsample
            x = upsample_add(transformed[-1 - idx], transformed[-idx], size)
            pyramids.append(func(x))

        pyramids = pyramids[::-1]  # reverse

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            # print('debug-----', x.size())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        if phase == 'eval':
            output = (loc, self.softmax(conf))
        else:
            output = (loc, conf)
        return output


def upsample_add(x, y, up_size):
    """Upsample and add two feature maps.
        Args:
          x: (Variable) lateral feature map.
          y: (Variable) top feature map to be upsampled.
          up_size: upsampled size
        Returns:
          (Variable) added feature map.(x + y*size)
    """
    if up_size is None:
        return x
    assert y.size(1) == x.size(1)  # same channel
    y = F.upsample(y, size=up_size, mode='bilinear')
    return x + y


# [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
def add_extras(layers_config):
    # Extra layers added to VGG for feature scaling
    extra_layers = []  # forward layers 3x3
    feature_transform_layers = []  # lateral layers 1x1
    pyramid_feature_layers = []  # final output layers 3x3

    in_d = None
    out_d = layers_config[1][-1]
    for layer, depth in zip(layers_config[0], layers_config[1]):
        if layer == 'S':
            extra_layers += [  # conv7_1 conv7_2 conv8_1 conv8_2
                nn.Conv2d(in_d, int(depth / 2), kernel_size=1),
                nn.Conv2d(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
            in_d = depth
        # feature map dimension is 5 or 3 when input is 300
        elif layer == '':  # conv9 conv10
            extra_layers += [
                nn.Conv2d(in_d, int(depth / 2), kernel_size=1),
                nn.Conv2d(int(depth / 2), depth, kernel_size=3)]
            in_d = depth
        else:  # use vgg layer
            in_d = depth
        feature_transform_layers += [nn.Conv2d(depth, out_d, kernel_size=1)]
        pyramid_feature_layers += [
            nn.Conv2d(out_d, out_d, kernel_size=3, padding=1)]
    return extra_layers, (feature_transform_layers, pyramid_feature_layers)


def multibox(transforms, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(transforms):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers



# extras_config = {  # conv4_3 relu, fc7 relu
#     'ssd': [[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]]  # feature map
# }


# def build(phase, cfg, base):
#     # extras, features_ = add_extras(extras_config['ssd'])
#     head = multibox(features_[1], cfg['num_priors'], cfg['num_classes'])
#     return FPN(phase, cfg, base, extras, head, features_)


if __name__ == '__main__':
    import os.path as osp
    from lib.utils.config import cfg, merge_cfg_from_file
    from lib.models import model_factory
    cfg_path = osp.join(cfg.GENERAL.CFG_ROOT, 'tests', 'test_train_fpn.yml')
    merge_cfg_from_file(cfg_path)

    net, priors, layer_dims = model_factory(phase='train', cfg=cfg)
    print(net)

    # print(priors)
    # print(layer_dims)

    # input_names = ['data']
    # net = net.cuda()
    # dummy_input = Variable(torch.randn(1, 3, 300, 300)).cuda()
    # torch.onnx.export(net, dummy_input, "./cache/alexnet.onnx", export_params=False, verbose=True, )
