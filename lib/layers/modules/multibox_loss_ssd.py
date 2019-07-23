# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.datasets import VARIANCE
# from ..box_utils import match_ssd, log_sum_exp


"""


class MultiBoxLossSSD(nn.Module):
    # criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
    #                             False, args.cuda)
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 min_neg_samples = 1, loc_weight=1.0, use_gpu=True):
        super(MultiBoxLossSSD, self).__init__()
        self.use_gpu = use_gpu
        self.loc_weight = loc_weight#Variable(torch.FloatTensor([loc_weight]).cuda, requires_grad=True)
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.min_neg_samples = min_neg_samples
        self.neg_overlap = neg_overlap
        self.variance = VARIANCE

    def forward(self, predictions, targets, loc_t=None, conf_t=None):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]   #Shape(num_priors,4)
        num_priors = (priors.size(0))   
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        if loc_t is None:
            loc_t = torch.Tensor(num, num_priors, 4)
            conf_t = torch.LongTensor(num, num_priors)

            for idx in range(num):
                #print('debug ----', idx, '\n', targets[idx][:, -1].datasets[0])
                if targets[idx][:, -1].data[0] < 0:    #target is -1, no objs
                    loc_t[idx] = torch.Tensor([[-1, -1, -1, -1]] * num_priors)
                    conf_t[idx] = torch.LongTensor([0] * num_priors)    #is background
                    continue
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                defaults = priors.data              #Shape(8732, 4)
                match_ssd(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0    #Shape(num, 8732)
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  #Shape(32, 8732, 4)
        loc_p = loc_data[pos_idx].view(-1, 4)   #predict [dx, dy, sx, sy], Shape(32, 8732, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)  #encoded offsets to learn   [tx, ty, tw, th]
        if loc_p.dim() != 0:
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)   #Shape(num*num_priors, 21)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) #softmax_loss, 

        #loss_c: Shape(num*num_priors, 1)
        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)   #sorted loss_c, Shape(num, num_priors)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, min=self.min_neg_samples, max=pos.size(1)-1) #num_neg
        
        neg = idx_rank < num_neg.expand_as(idx_rank)    #get neg with idx < num_neg

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return self.loc_weight * loss_l, loss_c
"""