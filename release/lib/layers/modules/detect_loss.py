# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp


def matching(targets, priors, threshold, variance, use_gpu=True, gpu_id=0, cfg=None):
    num_priors = (priors.size(0))
    num = len(targets)
    num_class = cfg.MODEL.NUM_CLASSES
    if use_gpu:
        with torch.cuda.device(gpu_id):
            loc_t = torch.cuda.FloatTensor(num, num_priors, 4)
            conf_t = torch.cuda.LongTensor(num, num_priors)
            conf_weights = torch.cuda.FloatTensor([cfg.LOSS.CONF_WEIGHT_RATIO[0]] + [1] * (num_class-1))
            mimic_label_t = torch.cuda.LongTensor(num, num_priors)
    else:
        loc_t = torch.FloatTensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        conf_weights = torch.cuda.FloatTensor([cfg.LOSS.CONF_WEIGHT_RATIO[0]] + [1] * (num_class-1))
        mimic_label_t = torch.LongTensor(num, num_priors)
    conf_t.fill_(0)
    for idx in range(num):
        if targets[idx][:, -1].data[0] < 0: continue
        
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        # # filter short edge < 8
        # x = truths[:, 2:4] - truths[:, 0:2]
        # y, _ = x.min(dim=1)
        # z = (y > 0.004).unsqueeze(-1).expand_as(truths)
        # t = truths[z]
        # if len(t.size()) == 0:
        #     #print('no target on image')
        #     continue
        # else:
        #     truths = t.view(-1, 4)
        match(threshold, truths, defaults, variance, labels,
              loc_t, conf_t, idx, use_gpu, gpu_id,mimic_label_t)

    loc_t = Variable(loc_t, requires_grad=False)
    conf_t = Variable(conf_t, requires_grad=False)
    conf_weights = Variable(conf_weights.unsqueeze(0).expand(len(cfg.GENERAL.NET_CPUS), 1, num_class), requires_grad=False)
    return loc_t, conf_t, conf_weights


class DetectLoss(nn.Module):
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
    """

    def __init__(self, cfg):
        super(DetectLoss, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.threshold = cfg.LOSS.OVERLAP_THRESHOLD
        self.background_label = cfg.LOSS.BACKGROUND_LABEL
        self.encode_target = cfg.LOSS.ENCODE_TARGET
        self.use_prior_for_matching = cfg.LOSS.PRIOR_FOR_MATCHING
        self.do_neg_mining = cfg.LOSS.MINING
        self.negpos_ratio = cfg.LOSS.NEGPOS_RATIO
        self.neg_overlap = cfg.LOSS.NEG_OVERLAP
        self.variance = cfg.MODEL.VARIANCE

    # @profile
    def forward(self, predictions, match_result, tb_writer=None):
        loc_data, conf_data = predictions
        loc_t, conf_t, conf_weights = match_result

        # loc_data, conf_data = predictions
        num = loc_data.size(0)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        #loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #NOTE sort by -p
        loss_c = F.softmax(batch_conf, -1)   #caffe calc softmax
        loss_c = loss_c[:, 0]   #background col
        loss_c = torch.mul(loss_c, -1)

        # Hard Negative Mining
        loss_c[pos] = -2  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)  # must be here...
        loss_c_sorted, loss_idx = loss_c.sort(1, descending=True)
        
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, min=3, max=pos.size(1)-1) #num_neg

        neg_min_indices = Variable(torch.arange(num).type_as(num_neg.data), requires_grad=False)
        neg_min_ious = loss_c_sorted[neg_min_indices, num_neg.squeeze_(1)]
        neg = loss_c > neg_min_ious.unsqueeze(1).expand_as(loss_c_sorted)

        # Confidence Loss Including Positive and Negative Examples
        mask_idx = (pos+neg).gt(0)
        pos_neg_idx = mask_idx.unsqueeze(2).expand_as(conf_data)
        conf_p = torch.masked_select(conf_data, pos_neg_idx).view(-1, self.num_classes)
        targets_weighted = torch.masked_select(conf_t, mask_idx)

        loss_c = F.cross_entropy(conf_p, targets_weighted, weight=conf_weights, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum()
        # loss_l /= N
        # loss_c /= N
        return loss_l, loss_c, N


class DetectLossPost(nn.Module):
    def __init__(self, cfg):
        super(DetectLossPost, self).__init__()
        self.weight_ratio = cfg.LOSS.WEIGHT_RATIO

    def forward(self, net_outputs, tb_writer=None):
        loss_l, loss_c, N = net_outputs
        N = N.data.sum()
        loss_l = loss_l.sum() / (N / self.weight_ratio[1])
        loss_c = loss_c.sum() / (N / self.weight_ratio[2])
        loss = (loss_l + loss_c).sum()
        return loss, (loss_l.data[0], loss_c.data[0])
