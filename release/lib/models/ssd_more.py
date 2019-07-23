import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *


class SSD_MORE(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, cfg, base):
        super(SSD_MORE, self).__init__()
        if phase != "train" and phase != "eval":
            raise Exception("ERROR: Input phase: {} not recognized".format(phase))
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        # self.priors = None
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.out = None

        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)  # TODO automate this
        self.L2Norm2 = L2Norm(256, 20)  # TODO automate this
        self.L2Norm3 = L2Norm(128, 20)  # TODO automate this

        extras = add_extras(cfg.MODEL.SSD.EXTRA_CONFIG, base)
        head = multibox(base, extras, cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        # if self.phase == 'eval':  # TODO add to config
        #     self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        for k in range(9):  # TODO make it configurable
            x = self.base[k](x)
        s = self.L2Norm3(x)
        sources.append(s)

        # apply vgg up to conv4_3 relu
        for k in range(9, 16):  # TODO make it configurable
            x = self.base[k](x)
        s = self.L2Norm2(x)
        sources.append(s)

        for k in range(16, 23):  # TODO make it configurable
            x = self.base[k](x)
        s = self.L2Norm(x)  # can replace batchnorm    nn.BatchNorm2d(x)#
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
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
            output = loc, conf
        return output


def add_extras(cfg_extra, base, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = base[-2].out_channels  # TODO make this configurable
    flag = False
    for idx, v in enumerate(cfg_extra):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg_extra[idx + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(base, extra_layers, num_priors, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [7, 14, 21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(base[v].out_channels,
                                 num_priors[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(base[v].out_channels,
                                  num_priors[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], len(vgg_source)):  # k start from 2
        loc_layers += [nn.Conv2d(v.out_channels, num_priors[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, num_priors[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers

#
# extras_config = {
#     'ssd': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
# }


if __name__ == '__main__':
    from lib.utils.config import cfg, merge_cfg_from_file
    from lib.models import model_factory
    import os.path as osp
    cfg_path = osp.join(cfg.GENERAL.CFG_ROOT, 'tests', 'test_ssd_more.yml')
    merge_cfg_from_file(cfg_path)

    net, priors, layer_dims = model_factory(phase='train', cfg=cfg)
    print(net)
    # print(priors)
    # print(layer_dims)

    # input_names = ['data']
    # net = net.cuda()
    # dummy_input = Variable(torch.randn(1, 3, 300, 300)).cuda()
    # torch.onnx.export(net, dummy_input, "./cache/alexnet.onnx", export_params=False, verbose=True, )
