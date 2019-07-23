# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.datasets import VARIANCE
from ..box_utils import match, log_sum_exp
from .repulsion_loss import RepulsionLoss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    e.g.
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh #0.5
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching  #True
        self.do_neg_mining = neg_mining  #True
        self.negpos_ratio = neg_pos #3
        self.neg_overlap = neg_overlap  #0.5
        self.variance = VARIANCE

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions   #tuple: a, b, c = ('a', 'b', 'c')
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        loc_g = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            predicts = loc_data[idx].data   #predict loc  Shape(8732, 4)
            truths = targets[idx][:, :-1].data  #gt coord   Shape(num_objs, 4)
            labels = targets[idx][:, -1].data   # gt label
            defaults = priors.data  #prior_box layer: default box   Shape(8732, 4)
            match(self.threshold, predicts, truths, defaults, self.variance, labels,
                  loc_t, loc_g, conf_t, idx)
            #loc_t, loc_g Shape(32, 8732, 4);    conf_t Shape(32, 8732)
            # print(predicts.size(), truths.size(), defaults.size(), '\n', \
            #     labels.size(), loc_t.size(), loc_g.size(), conf_t.size())
            
        if self.use_gpu:
            loc_t = loc_t.cuda()
            loc_g = loc_g.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        loc_g = Variable(loc_g, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        
        pos = conf_t > 0    #Shape(32, 8732)
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        #loc_data:(batch_size,num_priors,4)   #predict_loc
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)   #predict loc [dx, dy, sx, sy]
        loc_t = loc_t[pos_idx].view(-1, 4)  #[num*num_priors,4] encoded offsets to learn   [tx, ty, tw, th]
        loc_g = loc_g[pos_idx].view(-1, 4)  #prior_box with second largest IoU
        
        #priors = priors.unsqueeze(0).expand_as(pos_idx)    #
        priors = priors[pos_idx].view(-1, 4)    #Shape(num_priors,4)???????
        
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        repul_loss = RepulsionLoss(sigma=1.)#0.
        loss_l_repul = repul_loss(loc_p, loc_g, priors)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)   #[num_cells, 21]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) #for each img find num_neg in batch
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_l_repul /= N
        loss_c /= N
        return loss_l, loss_l_repul, loss_c
