"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import math
import numpy as np
#import caffe_pb2 as pb2
from caffe.proto import caffe_pb2 as pb2

def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob

def CopyTuple(param):
    if isinstance(param, tuple):
        return param
    elif isinstance(param, int):
        return param, param
    else:
        assert type(param)

def ty(caffe_type):
    def f(_):
        layer = pb2.LayerParameter()
        layer.type = caffe_type
        return layer
    return f

def data(inputs, use_cuda=True):
    layer = pb2.LayerParameter()
    layer.type = 'Input'
    input_shape = pb2.BlobShape()
    if use_cuda:
        input_shape.dim.extend(inputs.data.cpu().numpy().shape)
    else:
        input_shape.dim.extend(inputs.data.numpy().shape)
    layer.input_param.shape.extend([input_shape])
    return layer

def Slice(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Slice"

    layer.slice_param.axis = pytorch_layer.axis
    layer.slice_param.slice_point.extend(pytorch_layer.slice_point)
    return layer

def ShuffleChannel(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "ShuffleChannel"
    # group only set by manually
    layer.shuffle_channel_param.group = 3
    return layer

def inner_product(pytorch_layer, use_cuda=True):
    layer = pb2.LayerParameter()
    layer.type = "InnerProduct"

    if use_cuda:
        blobs_weight = pytorch_layer.next_functions[2][0].next_functions[0][0].variable.data.cpu().numpy()
    else:
        blobs_weight = pytorch_layer.next_functions[2][0].next_functions[0][0].variable.data.numpy()
    num_output = pytorch_layer.next_functions[2][0].next_functions[0][0].variable.size(0)
    layer.inner_product_param.num_output = num_output

    if pytorch_layer.next_functions[0][0]:
        layer.inner_product_param.bias_term = True
        bias = pytorch_layer.next_functions[0][0].next_functions[0][0].variable.data.cpu().numpy()
        layer.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer.inner_product_param.bias_term = False
        layer.blobs.extend([as_blob(blobs_weight)])

    return layer


def concat(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Concat"
    #TODO
    layer.concat_param.axis = 1#int(pytorch_layer.dim)
    return layer


def flatten(pytorch_layer):
    layer_flat = pb2.LayerParameter()
    layer_flat.type = 'Flatten'
    layer_flat.flatten_param.axis = 1
    return layer


def spatial_convolution(pytorch_layer):
    layer = pb2.LayerParameter()
    blobs_weight = pytorch_layer.next_functions[1][0].variable.data.cpu().numpy()
    assert len(blobs_weight.shape) == 4, blobs_weight.shape
    (nOutputPlane, nInputPlane, kH, kW) = blobs_weight.shape

    padH = pytorch_layer.padding[0]
    padW = pytorch_layer.padding[1]
    dH = pytorch_layer.stride[0]
    dW = pytorch_layer.stride[1]
    dilation = pytorch_layer.dilation[0]

    if pytorch_layer.transposed:    #F.conv_transpose is unuse. try nn.ConvTranspose2D
        layer.type = "Deconvolution"
        layer.convolution_param.num_output = nInputPlane
    else:
        layer.type = "Convolution"
        layer.convolution_param.num_output = nOutputPlane

    if kH == kW:
        layer.convolution_param.kernel_size.extend([kH])
    else:
        layer.convolution_param.kernel_h = kH
        layer.convolution_param.kernel_w = kW
    if dH == dW:
        layer.convolution_param.stride.extend([dH])
    else:
        layer.convolution_param.stride_h = dH
        layer.convolution_param.stride_w = dW
    if padH == padW:
        layer.convolution_param.pad.extend([padH])
    else:
        layer.convolution_param.pad_h = padH
        layer.convolution_param.pad_w = padW
    layer.convolution_param.dilation.extend([dilation])
    layer.convolution_param.group = pytorch_layer.groups

    if pytorch_layer.next_functions[2][0]:
        layer.convolution_param.bias_term = True
        bias = pytorch_layer.next_functions[2][0].variable.data.cpu().numpy()
        layer.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer.convolution_param.bias_term = False
        layer.blobs.extend([as_blob(blobs_weight)])

    return layer


def FillBilinear(ch, k):
    blob = np.zeros(shape=(ch, 1, k, k))

    """ Create bilinear weights in numpy array """
    bilinear_kernel = np.zeros([k, k], dtype=np.float32)
    scale_factor = (k + 1) // 2
    if k % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(k):
        for y in range(k):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)

    for i in range(ch):
        blob[i, 0, :, :] = bilinear_kernel
    return blob


def UpsampleBilinear(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Deconvolution"

    #print(dir(pytorch_layer), pytorch_layer.scale_factor)
    #assert pytorch_layer.scale_factor[0] == pytorch_layer.scale_factor[1]
    factor = int(pytorch_layer.scale_factor)
    c = int(pytorch_layer.output_channel)
    k = 2 * factor - factor % 2

    layer.convolution_param.num_output = c
    layer.convolution_param.kernel_size.extend([k])
    layer.convolution_param.stride.extend([factor])
    layer.convolution_param.pad.extend([int(math.ceil((factor - 1) / 2.))])
    layer.convolution_param.group = c
    layer.convolution_param.weight_filler.type = 'bilinear'
    layer.convolution_param.bias_term = False

    learning_param = pb2.ParamSpec()
    learning_param.lr_mult = 0
    learning_param.decay_mult = 0
    layer.param.extend([learning_param])

    """ Init weight blob of filter kernel """
    blobs_weight = FillBilinear(c, k)
    layer.blobs.extend([as_blob(blobs_weight)])

    return layer


def CopyPoolingParameter(pytorch_layer, layer):

    kH, kW = CopyTuple(pytorch_layer.kernel_size)
    dH, dW = CopyTuple(pytorch_layer.stride)
    padH, padW = CopyTuple(pytorch_layer.padding)

    if kH == kW:
        layer.pooling_param.kernel_size = kH
    else:
        layer.pooling_param.kernel_h = kH
        layer.pooling_param.kernel_w = kW
    if dH == dW:
        layer.pooling_param.stride = dH
    else:
        layer.pooling_param.stride_h = dH
        layer.pooling_param.stride_w = dW
    if padH == padW:
        layer.pooling_param.pad = padH
    else:
        layer.pooling_param.pad_h = padH
        layer.pooling_param.pad_w = padW

    if pytorch_layer.ceil_mode is True:
        return

    if pytorch_layer.ceil_mode is False:
        if padH == padW:
            if dH > 1 and padH > 0:
                layer.pooling_param.pad = padH - 1
        else:
            if dH > 1 and padH > 0:
                layer.pooling_param.pad_h = padH - 1
            if dW > 1 and padW > 0:
                layer.pooling_param.pad_w = padW - 1

def MaxPooling(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    layer.pooling_param.pool = pb2.PoolingParameter.MAX
    layer.pooling_param.kernel_size = 2
    layer.pooling_param.pad = 0
    layer.pooling_param.stride = 2
    #CopyPoolingParameter(pytorch_layer, layer) #not surpport this
    return layer


def AvgPooling(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    layer.pooling_param.pool = pb2.PoolingParameter.AVE
    layer.pooling_param.kernel_size = 7
    layer.pooling_param.pad = 0
    layer.pooling_param.stride = 1
    #CopyPoolingParameter(pytorch_layer, layer)
    return layer

def AdaptiveAvgPooling(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    layer.pooling_param.pool = pb2.PoolingParameter.AVE
    layer.pooling_param.global_pooling = True
    #CopyPoolingParameter(pytorch_layer, layer)
    return layer

def dropout(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Dropout"
    layer.dropout_param.dropout_ratio = float(pytorch_layer.p)
    train_only = pb2.NetStateRule()
    train_only.phase = pb2.TEST #TRAIN
    layer.exclude.extend([train_only])
    return layer


def elu(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "ELU"
    layer.elu_param.alpha = pytorch_layer.additional_args[0]
    return layer


def leaky_ReLU(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "ReLU"
    layer.relu_param.negative_slope = float(pytorch_layer.additional_args[0])
    return layer


def PReLU(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "PReLU"
    #TODO
    # print dir(pytorch_layer)
    # print pytorch_layer.saved_tensors, type(pytorch_layer.saved_tensors)
    # print '------[['
    # print pytorch_layer.next_functions[1][0].variable

    num_parameters = pytorch_layer.next_functions[1][0].variable.data.cpu().numpy().shape
    # print num_parameters
    layer.prelu_param.channel_shared = True if len(num_parameters) == 1 else False

    blobs_weight = pytorch_layer.next_functions[1][0].variable.data.cpu().numpy()[0]
    layer.blobs.extend([as_blob(blobs_weight)])
    return layer


def MulConst(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Power"
    layer.power_param.power = 1
    layer.power_param.scale = float(pytorch_layer.constant)
    layer.power_param.shift = 0
    return layer


def AddConst(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Power"
    layer.power_param.power = 1
    layer.power_param.scale = 1
    """ Constant to add should be filled by hand, since not visible in autograd """
    layer.power_param.shift = float('inf')
    return layer


def softmax(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = 'Softmax'
    return layer


def eltwise(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    return layer


def eltwise_max(pytorch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    layer.eltwise_param.operation = 2
    return layer


def batchnorm(pytorch_layer):
    layer_bn = pb2.LayerParameter()
    layer_bn.type = "BatchNorm"

    layer_bn.batch_norm_param.use_global_stats = 1
    layer_bn.batch_norm_param.eps = pytorch_layer.eps
    layer_bn.blobs.extend([
        as_blob(pytorch_layer.running_mean.cpu().numpy()),
        as_blob(pytorch_layer.running_var.cpu().numpy()),
        as_blob(np.array([1.]))
    ])

    layer_scale = pb2.LayerParameter()
    layer_scale.type = "Scale"

    blobs_weight = pytorch_layer.next_functions[1][0].variable.data.cpu().numpy()

    if pytorch_layer.next_functions[2][0]:
        layer_scale.scale_param.bias_term = True
        bias = pytorch_layer.next_functions[2][0].variable.data.cpu().numpy()
        layer_scale.blobs.extend([as_blob(blobs_weight), as_blob(bias)])
    else:
        layer_scale.scale_param.bias_term = False
        layer_scale.blobs.extend([as_blob(blobs_weight)])

    return [layer_bn, layer_scale]

def Scale(pytorch_layer):
    layer_scale = pb2.LayerParameter()
    layer_scale.type = "Scale"

    layer_scale.scale_param.bias_term = False
    layer_scale.scale_param.axis = 0

    return layer_scale


def Permute(pytorch_layer):
    layer_permute = pb2.LayerParameter()
    layer_permute.type = 'Permute'
    #print(type(layer_permute.permute_param.order), dir(layer_permute.permute_param.order))
    layer_permute.permute_param.order.extend([0,2,3,1])

    layer_flat = pb2.LayerParameter()
    layer_flat.type = 'Reshape'
    # print(dir(layer_flat.reshape_param), type(layer_flat.reshape_param.shape))
    layer_flat.reshape_param.shape.dim.extend([0,-1,1,1])
    return [layer_permute, layer_flat]

def sample_channel(pytorch_layer):
    layer_slice = pb2.LayerParameter()
    layer_slice.type = 'Slice'
    #print(type(layer_permute.permute_param.order), dir(layer_permute.permute_param.order))
    layer_slice.slice_param.axis = 1
    layer_slice.slice_param.slice_point.extend([1,])

    layer_concat = pb2.LayerParameter()
    layer_concat.type = "Concat"
    layer_concat.concat_param.axis = 1#int(pytorch_layer.dim)
    return [layer_slice, layer_concat]

def build_converter(opts):
    return {
        'data': data,
        'Addmm': inner_product,
        'Threshold': ty('ReLU'),
        'ConvNd': spatial_convolution,
        'MaxPool2D': MaxPooling,   #MaxPool2DBackward
        'AvgPool2D': AvgPooling,
        'AdaptiveAvgPool2d': AdaptiveAvgPooling,
        'Add1': eltwise,    #AddBackward1
        'Cmax': eltwise_max,
        'BatchNorm': batchnorm,
        'Mul1': Scale,  #SENet *scale
        'Cat': concat,    #CatBackward
        'Dropout': dropout,
        'UpsamplingBilinear2d': UpsampleBilinear,
        'MulConstant': MulConst,
        'AddConstant': AddConst,
        'Softmax': softmax,
        'Sigmoid0': ty('Sigmoid'),
        'Tanh0': ty('TanH'),
        'ELU': elu,
        'LeakyReLU': leaky_ReLU,
        'PReLU': PReLU,
        'Slice': Slice,
        'IndexSelect': sample_channel,
        'View': flatten,    #pass
        'Index': Slice,
        'Transpose': ShuffleChannel,
        'Permute': Permute,
        'Hardtanh': ty('ReLU6'),
    }

def convert_caffe(opts, typename, pytorch_layer):
    converter = build_converter(opts)
    if typename not in converter:
        raise ValueError("Unknown layer type: {}, known types: {}".format(
            typename, converter.keys()))
    return converter[typename](pytorch_layer)
