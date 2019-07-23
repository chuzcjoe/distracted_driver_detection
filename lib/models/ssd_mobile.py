import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *


class SSD_MOBILE(nn.Module):
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
        super(SSD_MOBILE, self).__init__()
        if phase != "train" and phase != "eval" and phase != "mimic" and phase != "caffe":
            raise Exception("ERROR: Input phase: {} not recognized".format(phase))
        self.phase = phase
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
        self.priors = None
        self.image_size = cfg.MODEL.IMAGE_SIZE
        for k in range(11):
            print(k,base[k])
        # SSD network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(512, 20)  # TODO automate this
        
        extras = add_extras(cfg.MODEL.SSD.EXTRA_CONFIG, base)

        head = multibox(base,cfg.MODEL.SSD.EXTRA_CONFIG, cfg.MODEL.NUM_PRIOR, cfg.MODEL.NUM_CLASSES)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        # self.matching = None
        self.criterion = None
        # if self.phase == 'eval':  # TODO add to config
        #     self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, phase='train', match_result=None, tb_writer=None):
        sources = list()
        loc = list()
        conf = list()

        phase = phase

        # dstf1 = open("/home/users/xupengfei/pytorch/ssd.pytorch/data.txt",'w')
        # dstf1.write(x)
        # print("x",x)
        # apply vgg up to conv4_3 relu
        for k in range(11):  # TODO make it configurable
            x = self.base[k](x)
            if k == 10 or k == 8 or k == 5:
                sources.append(x)


        # s = self.L2Norm(x)  # can replace batchnorm    nn.BatchNorm2d(x)#
        #sources.append(s)

        # apply vgg up to fc7
        # for k in range(7, len(self.base)):
        #     x = self.base[k](x)
        # sources.append(x)

        # apply extra layers and cache source layer outputs
        # for k, v in enumerate(self.extras):
        #     sources[k] = F.relu(v(sources[k]), inplace=True)
        headnum = len(self.extras)/2 #TODO head config 

        for k, v in enumerate(self.extras):
            idx = int(k//headnum)
            sources[idx] = v(sources[idx])
        # apply multibox head to source layers
        for (source, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(source).permute(0, 2, 3, 1).contiguous())
            conf.append(c(source).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        
        if phase == 'eval':
            output = loc, self.softmax(conf)
        elif phase == 'train':

            output = self.criterion((loc, conf), match_result, tb_writer) \
                if self.criterion is not None else None
            # output = loc, conf
        elif phase == 'caffe':
            output = (loc, conf)
        else:
            output = sources

        return output


def add_extras(cfg_extra, base, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels_p2 = 0
    in_channels_p3 = 0
    for idx, v in enumerate(cfg_extra):
        if 'p2' in v:
            blockidx = 5 
            if in_channels_p2 == 0:
                in_channels_p2 = base[blockidx].conv2.out_channels  # TODO make this configurable
            layers += [nn.Conv2d(in_channels_p2, int(v[3:]),kernel_size=1, stride=1)]
            layers += [nn.ReLU(inplace=True)]
            in_channels_p2 = int(v[3:])
        elif 'p3' in v:
            blockidx = 8 
            if in_channels_p3 == 0:
                in_channels_p3 = base[blockidx].conv2.out_channels  # TODO make this configurable
            layers += [nn.Conv2d(in_channels_p3, int(v[3:]),kernel_size=1, stride=1)]
            layers += [nn.ReLU(inplace=True)]
            in_channels_p3 = int(v[3:])
    return layers


def multibox(base, cfg,num_priors, num_classes):
    loc_layers = []
    conf_layers = []

    for k, v in enumerate(cfg):  
        if 'p2' in v:
            in_channel_p2 = int(v[3:])
        elif 'p3' in v:
            in_channel_p3 = int(v[3:])
    loc_layers += [nn.Conv2d(in_channel_p2, num_priors[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(in_channel_p2, num_priors[0]
                              * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(in_channel_p3, num_priors[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(in_channel_p3, num_priors[1]
                              * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(base[-1].conv2.out_channels, num_priors[-1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(base[-1].conv2.out_channels, num_priors[-1]
                              * num_classes, kernel_size=3, padding=1)]
    return loc_layers, conf_layers

#
# extras_config = {
#     'ssd': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
# }


if __name__ == '__main__':
    from lib.utils.config import cfg
    from lib.models import model_factory

    net, priors, layer_dims = model_factory(phase='train', cfg=cfg)
    print(net)
    # print(priors)
    # print(layer_dims)

    # input_names = ['data']
    # net = net.cuda()
    # dummy_input = Variable(torch.randn(1, 3, 300, 300)).cuda()
    # torch.onnx.export(net, dummy_input, "./cache/alexnet.onnx", export_params=False, verbose=True, )
