"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import torch
from torch.autograd import Variable


def link_caffe(layer, name, bottom, top):
    layer.name = name
    for b in bottom:
        layer.bottom.append(b)
    for t in top:
        layer.top.append(t)

    caffe_net.append(layer)


def link_ncnn(layer, name, bottom, top):
    pass

def GetLayerParam_Index(func):
    for axis, slice_param in enumerate(func.index):
        if isinstance(slice_param, int):
            start = slice_param
            stop = slice_param + 1
        else:
            start = slice_param.start
            stop = slice_param.stop
            step = slice_param.step
        if (start or stop or step) is not None:
            break
    shape = func.input_size
    dim_size = shape[axis]
    return start, stop, dim_size, axis

bottom_weights = torch.rand(1,1,1,1)
def DFS(func):
    if func in visited:
        return tops_dict[func]
    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    father_func = None
    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward') and\
                    (layer_type != 'AddmmBackward' or (child_type != 'ExpandBackward' and child_type != 'TBackward')):
                    child_name = DFS(u[0])
                    bottoms.append(child_name)
                    father_func = u[0]

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """ Reaching the root node """
    """  TODO: multi data input """
    if len(bottoms) == 0:
        if 'data' not in layer_type_count:
            layer_type_count['data'] = 1
            """ Gen data layer """
            layer_data = convert('', 'data', inputs)
            link(layer_data, 'data', [], ['data'])

        """ Link it with data input """
        bottoms.append('data')

    """  Skip some pytorch layers  """
    if dst == 'caffe':
        if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem']:
            tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            tops_dict[func] = bottoms[0]
        else:
            tops_dict[func] = name

    """ Split to BatchNorm and Scale """
    if layer_type_name == 'BatchNorm':
        layer_double = convert('', layer_type_name, func)
        scale_name = name + '_' + 'scale'
        link(layer_double[0], name, bottoms, [tops_dict[func]])
        link(layer_double[1], scale_name, [tops_dict[func]], [tops_dict[func]])
     
    # add Flatten layer in back of Permute 
    elif layer_type_name == 'Permute':
        layer_double = convert('', layer_type_name, func)
        flatten_name = name + '_' + 'flat'
        if dst == 'caffe':
            link(layer_double[0], name+'_permute', bottoms, [tops_dict[func]+'_permute'])
            link(layer_double[1], flatten_name,  [tops_dict[func]+'_permute'], [tops_dict[func]])
    
    #special for Deconvlution
    elif layer_type_name == 'ConvNd':#str(type(father_func).__name__) == 'CudaTransferBackward' and layer_type_name == 'ConvNd':
        #global bottom_weights
        try: #normal convlution layer
            weights = func.next_functions[1][0].variable    #use for try
            layer = convert('', layer_type_name, func)
            link(layer, name, bottoms, [tops_dict[func]])
        except: #UpsamplingBilinear2d
            try:
                weights = func.next_functions[0][0].next_functions[1][0].variable   #bottom of Deconvlution layer is convlution layer
                #weights = func.next_functions[0][0].next_functions[0][0].running_mean  #bottom of Deconvlution layer is batchnorm layer
                #weights = func.next_functions[0][0].next_functions[0][0].next_functions[1][0].variable #2th bottom of Deconvlution layer is batchnorm layer
            except AttributeError:
                weights = torch.rand(512,1,1,1)
            deconv_func = torch.autograd.function    # add extra attributes
            deconv_func.output_channel = weights.size(0)
            #TODO # scale_factor only set by manually
            deconv_func.scale_factor = 2
            layer = convert('', 'UpsamplingBilinear2d', deconv_func)
            link(layer, name, [tops_dict[func.next_functions[0][0]]], [tops_dict[func]])
    # IndexSelect is combined by slice and concat
    elif layer_type_name == 'IndexSelect':
        layer_double = convert('', layer_type_name, func)
        concat_name = name + '_cat'
        if dst == 'caffe':
            link(layer_double[0], name+'_slice', bottoms, [tops_dict[func]+'_slice'])
            link(layer_double[1], concat_name,  [tops_dict[func]+'_slice'], [tops_dict[func]])
    # others
    elif layer_type_name not in ['Index', 'Clone', 'SetItem', 'View', 'CudaTransfer', 'Mul0', 'Expand']:
        # print str((type(func).__name__)), layer_type_name, len(func.next_functions)
        layer = convert('', layer_type_name, func)
        link(layer, name, bottoms, [tops_dict[func]])

    #skip some layer
    if layer_type_name == 'Index':
        """ Change layer name for 'Slice' """
        tops_dict[func] = tops_dict[father_func] + '_' + tops_dict[func]
    elif layer_type_name == 'View' or layer_type_name == 'Mul0' or layer_type_name == 'Expand':
        """ Change layer name for 'View' """
        #father_func_name = str(type(father_func).__name__)
        tops_dict[func] = tops_dict[father_func]

    """ If func layer has multiple top layers """
    if (func in multi_tops) and (len(multi_tops[func]) > 1):
        if func in slice_point:
            """ Make an extra dummy layer type 'Slice' after func layer, which not exist in pytorch """
            slice_func = torch.autograd.function    # add extra attributes
            slice_func.axis = axis_dict[func]
            slice_func.slice_point = slice_point[func]
            slice_layer = convert('', 'Slice', slice_func)
            #####
            new_name = [] 
            for _layer_name in multi_tops[func]:
                new_name.append(_layer_name.replace(name, tops_dict[father_func]))
            link(slice_layer, tops_dict[father_func] + '_slicer', [tops_dict[father_func]], new_name)

    return tops_dict[func]


def FindMultiTops(func):
    """
        Precount nodes with number of tops(indegree)>1,
        which could be Slice or Split(only in ncnn, for making multiple copies)
    """
    if func in visited:
        return tops_dict[func]

    visited.add(func)
    layer_type = str(type(func).__name__)
    bottoms = []

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward') and\
                    (layer_type != 'AddmmBackward' or (child_type != 'ExpandBackward' and child_type != 'TBackward')):
                    child_name = FindMultiTops(u[0])
                    bottoms.append(child_name)
                    #father_func = u[0]

    """ Gen layer name """
    layer_type_name = layer_type.replace('Backward', '')
    if layer_type_name in layer_type_count:
        layer_type_count[layer_type_name] += 1
    else:
        layer_type_count[layer_type_name] = 1

    name = layer_type_name + '_' + str(layer_type_count[layer_type_name])

    """  Skip some pytorch layers  """
    if dst == 'caffe':
        if layer_type_name in ['Clone', 'Threshold', 'Dropout', 'SetItem']:
            tops_dict[func] = bottoms[0]
        elif (layer_type_name == 'Index') and (not isinstance(func.index, tuple)):
            tops_dict[func] = bottoms[0]
        else:
            tops_dict[func] = name

    if hasattr(func, 'next_functions'):
        for u in func.next_functions:
            if u[0] is not None:
                child_type = str(type(u[0]).__name__)
                if child_type != 'AccumulateGrad' and (layer_type != 'AddmmBackward' or child_type != 'TransposeBackward') and\
                    (layer_type != 'AddmmBackward' or (child_type != 'ExpandBackward' and child_type != 'TBackward')):
                    father_func = u[0]
                    if father_func not in multi_tops:
                        multi_tops[father_func] = []
                    multi_tops[father_func].append(tops_dict[father_func] + '_' + tops_dict[func])

                    if (layer_type == 'IndexBackward') and isinstance(func.index, tuple):
                        if father_func not in slice_point:
                            slice_point[father_func] = []
                        start, stop, dim_size, axis = GetLayerParam_Index(func)

                        """ Persume the visit of Index layers will be ascending """
                        if start > 0:
                            slice_point[father_func].append(start)
                            axis_dict[father_func] = axis

                            """ Last slice """
                            # if stop == dim_size

    return tops_dict[func]

def ConvertModel_caffe(pytorch_net, InputShape, softmax=False, use_cuda=False, save_graph=False):
    """ Pytorch to Caffe, only support single tensor input """
    import os
    #import caffe_pb2 as pb2
    from caffe.proto import caffe_pb2 as pb2
    from ConvertLayer_caffe import convert_caffe

    """ Need forward once """
    global inputs
    n, c, h, w = InputShape
    if use_cuda:
        print('Convert in cuda...')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        pytorch_net = pytorch_net.cuda()
        inputs = Variable(torch.zeros(n, c, h, w).cuda(), requires_grad=True)
    else:
        inputs = Variable(torch.zeros(n, c, h, w), requires_grad=True)
    
    pytorch_net.eval()
    outputs = pytorch_net(inputs, phase='eval') #phase='eval'
    print('model outputs:')
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        #print(outputs[0].shape) #(1, 10830 4)
        print(outputs[0][-1])   #last loc
    else:
        print(outputs)

    if save_graph:
        from visualize import make_dot
        model_name = str(pytorch_net.__class__.__name__) + '_caffe'
        fp = open("{}.dot".format(model_name), "w")
        dot = make_dot(outputs)
        print >> fp, dot
        fp.close()
    
    if softmax:
        import torch.nn as nn
        regularize = nn.Softmax()
        outputs = regularize(outputs)

    """ Travel computational graph in backward order """
    """ Need to count number of tops(indegree) of all nodes first """
    global visited, tops_dict, layer_type_count, dst
    global slice_point, multi_tops, axis_dict
    visited = set()
    tops_dict = dict()
    layer_type_count = dict()
    slice_point = dict()
    multi_tops = dict()
    axis_dict = dict()
    dst = 'caffe'

    for out in outputs:
        FindMultiTops(out.grad_fn)

    """ Travel computational graph in backward order """
    global caffe_net
    global convert, link
    convert = convert_caffe
    link = link_caffe
    caffe_net = []

    visited = set()
    tops_dict = dict()
    layer_type_count = dict()

    for out in outputs:
        DFS(out.grad_fn)

    """ Caffe input """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True

    """ Caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for layer in caffe_net:
        binary_weights.layer.extend([layer])

        layer_proto = pb2.LayerParameter()
        layer_proto.CopyFrom(layer)
        del layer_proto.blobs[:]
        text_net.layer.extend([layer_proto])

    return text_net, binary_weights