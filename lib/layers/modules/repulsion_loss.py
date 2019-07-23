# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
# from lib.datasets import VARIANCE
from ..box_utils import IoG, decode_new


class RepulsionLoss(nn.Module):

    def __init__(self, sigma=0., use_cuda=True):
        super(RepulsionLoss, self).__init__()
        self.use_cuda = use_cuda
        self.variance = [0.1, 0.2]

        self.sigma = Variable(torch.FloatTensor([sigma]), requires_grad=False)

        if use_cuda:
            self.sigma = self.sigma.cuda()

    #repul_loss(loc_p, loc_g, priors)
    def forward(self, loc_data, ground_data, prior_data):
        
        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        
        iog = IoG(ground_data, decoded_boxes)
        
        loss_repgt = smoothln(iog, self.sigma, self.use_cuda)

        #print ('loss_repgt', loss_repgt.size()) 
        return loss_repgt

def smoothln(x, sigma, use_cuda = True):    #need to test. test success
    loss_repgt = Variable(torch.FloatTensor([0.]), requires_grad=True)
    if use_cuda:
        loss_repgt = loss_repgt.cuda()

    mask1 = x.ge(sigma)   # >= sigma return 1 else 0
    term1 = torch.masked_select(x, mask1)  #Shape(num*num_priors)
    if term1.dim() != 0:
        term1 =  -torch.log(1 - term1 + 1e-10)
        loss_repgt += torch.sum(term1)

    mask2 = x.lt(sigma)   # < sigma return 1 else 0
    term2 = torch.masked_select(x, mask2)  #Shape(num*num_priors)
    if term2.dim() != 0:
        term2 =  (term2 - sigma) / (1. - sigma + 1e-10) - torch.log(1. - sigma + 1e-10)
        loss_repgt += torch.sum(term1)
    
    return loss_repgt