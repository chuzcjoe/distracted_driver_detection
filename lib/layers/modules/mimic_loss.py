# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp


def matching_mimic(targets, priors, threshold, variance, use_gpu=True, gpu_id=0, cfg=None):
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
        mimic_label_t = torch.cuda.LongTensor(num, num_priors)
    conf_t.fill_(0)
    mimic_label_t.fill_(0)

    for idx in range(num):
        if targets[idx][:, -1].data[0] < 0: continue
        
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data

        match(threshold, truths, defaults, variance, labels,
              loc_t, conf_t, idx, use_gpu, gpu_id,mimic_label_t)
    # for i in range(mimic_label_t.size()[0]):
    #     for j in range(mimic_label_t.size()[1]):
    #         # print(mimic_label[i,j])
    #         if mimic_label_t[i,j] > 0 :
    #             print("11111   ",priors[j,:])

    # print('priors_mimic_t',len(priors_mimic_t))
    # print('priors_mimic_t',len(priors_mimic_t[0]))


    mimic_label_t = Variable(mimic_label_t, requires_grad=False)
    # conf_t = Variable(conf_t, requires_grad=False)
    # conf_weights = Variable(conf_weights.unsqueeze(0).expand(len(cfg.GENERAL.NET_CPUS), 1, num_class), requires_grad=False)
    return mimic_label_t


class MimicLoss(nn.Module):
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
        super(MimicLoss, self).__init__()
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
    def forward(self, map_t, map_s,mimic_label,priors):
        # print("type(map_t)",type(map_t))
        # print("type(map_s)",type(map_s))
        # print("type(map_t)",map_t[2].size())
        # print("type(map_s)",map_s[2].size())
        #[32, 96, 90, 160],([32, 96, 45, 80]),[32, 96, 23, 40]
        # print("type(mimic_label)",type(mimic_label))
        # print("type(priors)",type(priors))   
        # print("type(mimic_label)",mimic_label.size())
        # print("priors",priors.size())
        # print(mimic_label)

        defaults = priors.data
        loss = 0
        for i in range(mimic_label.size()[0]):
            tmp_t = torch.cuda.FloatTensor(map_t[2].size(1),map_t[2].size(2), map_t[2].size(3))
            tmp_t.fill_(0)
            tmp_t = Variable(tmp_t, requires_grad=False)
            tmp_s = torch.cuda.FloatTensor(map_t[2].size(1),map_t[2].size(2), map_t[2].size(3))
            tmp_s.fill_(0)
            tmp_s = Variable(tmp_s)
            mask = torch.cuda.FloatTensor(map_t[2].size(2), map_t[2].size(3))
            mask.fill_(0)
            mask = Variable(mask, requires_grad=False)
            for j in range(mimic_label.size()[1]):
                # print(type(mimic_label.data[i,j]))
                if mimic_label.data[i,j] > 0:
                    x_min,y_min = (defaults[j, :2] - defaults[j, 2:]/2)
                    x_max,y_max = (defaults[j, :2] + defaults[j, 2:]/2)
                    x_min = max(0,int(x_min*map_t[2].size(3)))
                    y_min = max(0,int(y_min*map_t[2].size(2)))
                    x_max = min(int(x_max*map_t[2].size(3)),map_t[2].size(3))
                    y_max = min(int(y_max*map_t[2].size(2)),map_t[2].size(2))
                    for x_idx in range(x_min,x_max):
                        for y_idx in range(y_min,y_max):
                            mask.data[y_idx,x_idx]=1
            if mask.sum().data[0] == 0.0:
                loss = 0
                continue
            for c in range(map_t[2].size(1)):
                tmp_t[c]=map_t[2][i,c,].mul(mask)
                tmp_s[c]=map_s[2][i,c,].mul(mask)
            loss += (tmp_s - tmp_t).pow(2).sum()/mask.sum().data[0]/map_t[2].size(1)



        loss_mimic = loss/mimic_label.size()[0]
        return loss_mimic


class MimicLossPost(nn.Module):
    def __init__(self, cfg):
        super(MimicLossPost, self).__init__()
        self.weight_ratio = cfg.LOSS.WEIGHT_RATIO

    def forward(self, net_outputs, mimic_outputs,tb_writer=None):
        loss_l, loss_c, N = net_outputs
        N = N.data.sum()
        loss_l = loss_l.sum() / (N / self.weight_ratio[1])
        loss_c = loss_c.sum() / (N / self.weight_ratio[2])
        loss_mimic = mimic_outputs
        loss = (loss_l + loss_c + loss_mimic*10).sum()
        return loss, (loss_l.data[0], loss_c.data[0],loss_mimic)
