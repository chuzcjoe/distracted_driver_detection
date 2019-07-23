from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np
import cv2


def vis(func):
    """tensorboard visualization if has writer as input"""

    def wrapper(*args, **kw):
        return func(*args, **kw) if kw['tb_writer'] is not None else None

    return wrapper


class PriorBoxBase(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBoxBase, self).__init__()
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self._steps = cfg.MODEL.STEPS
        self._cfg_list = []
        self._prior_cfg = {}
        self._clip = cfg.MODEL.CLIP
        self._variance = cfg.MODEL.VARIANCE
        for v in self._variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def _setup(self, cfg):
        num_feat = len(self._steps)
        for item in self._cfg_list:
            if item not in cfg.MODEL:
                raise Exception("wrong anchor config!")
            if len(cfg.MODEL[item]) != num_feat and len(cfg.MODEL[item]) != 0:
                raise Exception("config {} length does not match step length!".format(item))
            self._prior_cfg[item] = cfg.MODEL[item]

    @property
    def num_priors(self):
        """allow prior num calculation before knowing feature map size"""
        assert self._prior_cfg is not {}
        return [int(len(self._create_prior(0, 0, k)) / 4) for k in range(len(self._steps))]

    def _create_prior(self, cx, cy, k):
        raise NotImplementedError

    @vis
    def _image_proc(self, image=None, tb_writer=None):
        # TODO test with image
        if isinstance(image, type(None)):
            image = np.ones((self.image_size[1], self.image_size[0], 3))
        elif isinstance(image, str):
            image = cv2.imread(image, -1)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        return image

    @vis
    def _prior_vis(self, anchor, image_ori, feat_idx, tb_writer=None):
        # TODO add output path to the signature
        writer = tb_writer.writer
        prior_num = self.num_priors[feat_idx]

        # transform coordinates
        scale = [self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]]
        bboxs = np.array(anchor).reshape((-1, 4))
        box_centers = bboxs[:, :2] * scale[:2]  # [x, y]
        # bboxs: [xmin, ymin, xmax, ymax]
        bboxs = np.hstack((bboxs[:, :2] - bboxs[:, 2:4] / 2, bboxs[:, :2] + bboxs[:, 2:4] / 2)) * scale
        box_centers = box_centers.astype(np.int32)
        bboxs = bboxs.astype(np.int32)
        # visualize each anchor box on a feature map
        for prior_idx in range(prior_num):
            image = image_ori.copy()
            bboxs_ = bboxs[prior_idx::prior_num, :]
            box_centers_ = box_centers[4 * prior_idx::prior_num, :]
            for archor, bbox in zip(box_centers_, bboxs_):
                cv2.circle(image, (archor[0], archor[1]), 1, (0, 0, 255), -1)
                if archor[0] == archor[1]:  # only show diagnal anchors
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            image = image[..., ::-1]
            writer.add_image('base/feature_map_{}_{}'.format(feat_idx, prior_idx), image, 2)

    def forward(self, layer_dims, tb_writer=None, image=None):
        priors = []
        image = self._image_proc(image=image, tb_writer=tb_writer)

        for k in range(len(layer_dims)):
            prior = []
            for i, j in product(range(layer_dims[k][0]), range(layer_dims[k][1])):
                steps_x = self.image_size[1] / self._steps[k]
                steps_y = self.image_size[0] / self._steps[k]
                cx = (j + 0.5) / steps_x  # unit center x,y
                cy = (i + 0.5) / steps_y
                prior += self._create_prior(cx, cy, k)
            priors += prior
            self._prior_vis(prior, image, k, tb_writer=tb_writer)

        output = torch.Tensor(priors).view(-1, 4)
        # TODO this clip is meanless, should clip on [xmin, ymin, xmax, ymax]
        if self._clip:
            output.clamp_(max=1, min=0)
        return output


class PriorBoxSSD(PriorBoxBase):
    def __init__(self, cfg):
        super(PriorBoxSSD, self).__init__(cfg)
        # self.image_size = cfg['image_size']
        self._cfg_list = ['MIN_SIZES', 'MAX_SIZES', 'ASPECT_RATIOS']
        self._flip = cfg.MODEL.FLIP
        self._setup(cfg)

    def _create_prior(self, cx, cy, k):
        # as the original paper do
        prior = []
        min_sizes = self._prior_cfg['MIN_SIZES'][k]
        min_sizes = [min_sizes] if not isinstance(min_sizes, list) else min_sizes
        for ms in min_sizes:
            # min square
            s_i = ms / self.image_size[0]
            s_j = ms / self.image_size[1]
            prior += [cx, cy, s_j, s_i]
            # min max square
            if len(self._prior_cfg['MAX_SIZES']) != 0:
                assert type(self._prior_cfg['MAX_SIZES'][k]) is not list  # one max size per layer
                s_i_prime = sqrt(s_i * (self._prior_cfg['MAX_SIZES'][k] / self.image_size[0]))
                s_j_prime = sqrt(s_j * (self._prior_cfg['MAX_SIZES'][k] / self.image_size[1]))
                prior += [cx, cy, s_j_prime, s_i_prime]
            # rectangles by min and aspect ratio
            for ar in self._prior_cfg['ASPECT_RATIOS'][k]:
                prior += [cx, cy, s_j * sqrt(ar), s_i / sqrt(ar)]  # a vertical box
                if self._flip:
                    prior += [cx, cy, s_j / sqrt(ar), s_i * sqrt(ar)]
        return prior


# PriorBox = PriorBoxSSD


def test_no_vis(cfg, tb_writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['min_sizes'] = [[30], [60], 111, 162, 213, 264]
    cfg['flip'] = True
    feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    p = PriorBoxSSD(cfg)
    print(p.num_priors)
    p1 = p.forward(feat_dim)
    print(p1)


def test_filp(cfg, tb_writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['flip'] = True
    feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    p = PriorBoxSSD(cfg)
    p1 = p.forward(feat_dim, tb_writer=tb_writer)

    cfg['flip'] = False
    cfg['aspect_ratios'] = [[2, 1 / 2], [2, 1 / 2, 3, 1 / 3], [2, 1 / 2, 3, 1 / 3],
                            [2, 1 / 2, 3, 1 / 3], [2, 1 / 2], [2, 1 / 2]]
    p = PriorBox(cfg)
    p2 = p.forward(feat_dim, tb_writer=tb_writer)
    # print(p2)
    assert (p2 - p1).sum() < 1e-8


def test_rectangle(cfg, tb_writer):
    cfg = copy.deepcopy(cfg)
    cfg['feature_maps'] = [38, 19, 10, 5, 3, 1]
    cfg['min_sizes'] = [30, 60, 111, 162, 213, 264]
    cfg['flip'] = True
    # feat_dim = [list(a) for a in zip(cfg['feature_maps'], cfg['feature_maps'])]
    # cfg['image_size'] = [300, 300]
    # feat_dim = [list(a) for a in zip(cfg['feature_maps'], [item * 2 for item in cfg['feature_maps']])]
    # cfg['image_size'] = [300, 600]
    feat_dim = [list(a) for a in zip([item * 2 for item in cfg['feature_maps']], cfg['feature_maps'])]
    cfg['image_size'] = [600, 300]
    p = PriorBoxSSD(cfg)
    p1 = p.forward(feat_dim, tb_writer=tb_writer)
    print(p1.shape)


if __name__ == '__main__':
    import copy
    # from lib.datasets.config import ssd_voc_vgg as cfg
    # from lib.utils.visualize_utils import TBWriter
    # tb_writer = TBWriter(log_dir, {'epoch': 50})
    #
    # test_no_vis(cfg, tb_writer)
    # test_filp(cfg, tb_writer)
    # test_rectangle(cfg, tb_writer)
    print('haha')
    from lib.utils.config import cfg
    print(cfg)

