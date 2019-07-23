# -*- coding: utf-8 -*-  
import torch
import os, sys
import time
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import cv2
import numpy as np
import argparse
from importlib import import_module
from torchvision import models

CUDA_VISIBLE_DEVICES="0"  #Specified GPUs range
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def load_filtered_state_dict(model, snapshot=None):
    # By user apaszke from discuss.pytorch.or
    model_dict = model.state_dict()
    # snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    new_snapshot = {}
    for k, v in snapshot.items():
        k = k.replace('module.', '')    #module is saved model with paralle mode  
        if k in model_dict:
            new_snapshot[k] = v
    model.load_state_dict(new_snapshot)

def get_flops(net, input_shape=(1, 3, 300, 300)):
    from flops_benchmark import add_flops_counting_methods, start_flops_count
    input = torch.ones(input_shape)
    input = torch.autograd.Variable(input)

    net = add_flops_counting_methods(net)
    net = net.train()
    net.start_flops_count()

    _ = net(input)

    return net.compute_average_flops_cost()/1e9/2

HOME = os.path.expanduser("~")
dataset_mean = (104, 117, 123)

#TODO user also can add 'w h' params
if len(sys.argv) < 3:
    print("Hint : python run.py torch_net_name.py torch_model_path c h w")
moudle = sys.argv[1]
trained_model = sys.argv[2]

c, h, w = 3, 224, 224
if len(sys.argv) == 6:
    c = int(sys.argv[3])
    h = int(sys.argv[4])
    w = int(sys.argv[5])
input_shape = (1, c, h, w)  #n c h w


net_file_path = os.path.abspath(moudle)
moudle_path = os.path.dirname(net_file_path) #without basename
net_moudle = os.path.splitext(os.path.basename(net_file_path))[0]#os.path.basename(net_file_path).replace('.py', '')

model_file_path = os.path.abspath(trained_model) 
save_path = os.path.dirname(model_file_path) + '/' #without basename


sys.path.append(moudle_path) #add net.py sys path
print moudle_path
torch_net = import_module(net_moudle)

#TODO
net = torch_net.build_model()
# net = models.resnet50(pretrained=False)
# net.fc = torch.nn.Linear(2048, 2)

# print(net.keys())

model_name = os.path.splitext(os.path.basename(model_file_path))[0]#str(net.__class__.__name__) + '_caffe'
net.eval()
if trained_model != 'None':
    try:
        load_filtered_state_dict(net, snapshot=torch.load(trained_model, map_location=lambda storage, loc: storage)['state_dict'])
        print('load state_dict...')
    except KeyError:
        load_filtered_state_dict(net, snapshot=torch.load(trained_model, map_location=lambda storage, loc: storage))
        print('load 2...')
    except TypeError:
        net = torch.load(trained_model, map_location=lambda storage, loc: storage)
        print('load 3...')
        # print type(weights), dir(weights)
        # net.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))
    except ImportError: # or AttributeError:
        net = torch.load(trained_model)
        print('load 4...')
    print('load model sucess!')
#net = net.cuda()


#net_flops = get_flops(net, input_shape)
#print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(net_flops))

##########onnx
# import torch.onnx
# input_var = Variable(torch.rand(input_shape))
# torch.onnx.export(net, input_var, "test.onnx", verbose=True)
################ onnx end

#save grad graph
# from visualize import make_dot
# output_var = net(input_var, 'eval')
# fp = open("{}.dot".format(model_name), "w")
# dot = make_dot(output_var)
# print >> fp, dot
# fp.close()

use_cuda = False

from ConvertModel import ConvertModel_caffe
print('Converting...')

text_net, binary_weights = ConvertModel_caffe(net, input_shape, softmax=False, use_cuda=use_cuda, save_graph=True)

import google.protobuf.text_format
print('save path:', save_path, model_name)
with open(save_path + model_name + '.prototxt', 'w') as f:
    f.write(google.protobuf.text_format.MessageToString(text_net))
with open(save_path + model_name + '.caffemodel', 'w') as f:
    f.write(binary_weights.SerializeToString())

