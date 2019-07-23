import torch
from torch import nn as nn
from torch.autograd import Variable

from .resnet import res10,res10_t
from .mobilenet import mobilenet
from .vgg import vgg16, tiny_vgg16, tiny_vgg16_p3, vgg16_reducedfc
from .ssd import SSD
from .ssd_sn import SSD_SN
from .ssd_mobile import SSD_MOBILE
from .ssd_mimic import SSD_MIMIC
from .ssd_more import SSD_MORE
from .ssd_coco import SSD_COCO
from .fpn import FPN
from .fssd import FSSD
from .rfb import RFB
from lib.layers import PriorBoxSSD

bases_list = ['vgg16','res10','res10_t','mobilenet', 'vgg16','vgg16_reducedfc', 'tiny_vgg16', 'tiny_vgg16_p3']
ssds_list = ['SSD', 'SSD_SN','SSD_MOBILE',  'SSD_MIMIC', 'FSSD', 'FPN', 'SSD_COCO', 'SSD_MORE', 'RFB']
priors_list = ['PriorBoxSSD']


def create(n, lst, **kwargs):
    if n not in lst:
        raise Exception("unkown type {}, possible: {}".format(n, str(lst)))
    return eval('{}(**kwargs)'.format(n))


def model_factory(phase, cfg):
    prior = create(cfg.MODEL.PRIOR_TYPE, priors_list, cfg=cfg)
    cfg.MODEL.NUM_PRIOR = prior.num_priors
    base = create(cfg.MODEL.BASE, bases_list)
    model = create(cfg.MODEL.TYPE, ssds_list, phase=phase, cfg=cfg, base=base)

    layer_dims = get_layer_dims(model, cfg.MODEL.IMAGE_SIZE)
    priors = prior.forward(layer_dims)
    return model, priors, layer_dims


def get_layer_dims(model, image_size):
    def forward_hook(self, input, output):
        """input: type tuple, output: type Variable"""
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.datasets.size(), output.datasets.norm()))
        dims.append([input[0].size()[2], input[0].size()[3]])  # h, w

    dims = []
    handles = []
    for idx, layer in enumerate(model.loc.children()):  # loc...
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(forward_hook)
            handles.append(hook)

    #input_size = (1, 1, image_size[0], image_size[1])
    input_size = (1, 3, image_size[0], image_size[1])
    model.eval()  # fix bn bugs
    model(Variable(torch.randn(input_size)), phase='eval')
    [item.remove() for item in handles]
    return dims
