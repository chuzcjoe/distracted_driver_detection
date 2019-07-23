# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

import collections

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"

    #assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    
    loc_layers = []
    conf_layers = []

    priorbox_layers = collections.OrderedDict()
    norm_name_layers = collections.OrderedDict()
    for i in range(0, num):
        from_layer = from_layers[i]
        # Get the normalize value.
        if normalizations:    #normalizations = [20, -1, -1, -1, -1, -1]
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                norm_name_layers[norm_name] = net.layer.add()
                norm_name_layers[norm_name].CopyFrom(L.Normalize(scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False).to_proto().layer[0])
                norm_name_layers[norm_name].name = norm_name
                norm_name_layers[norm_name].top[0] = norm_name
                norm_name_layers[norm_name].bottom.append(from_layer)
                from_layer = norm_name
    
        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        priorbox_layers[name] = net.layer.add()
        # priorbox_layers[name].CopyFrom(L.PriorBox(min_size=min_size, max_size=max_size, aspect_ratio=aspect_ratio, 
        #                                 step=step, flip=flip, clip=clip, variance=prior_variance, offset=offset).to_proto().layer[0])

        p = L.PriorBox(min_size=min_size, max_size=max_size, aspect_ratio=aspect_ratio, 
                                         step=step, flip=flip, clip=clip, variance=prior_variance, offset=offset)

        p1 = L.PriorBox(min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset).to_proto().layer[0]

        c = L.Convolution(kernel_size=7, stride=1, num_output=48, pad=0).to_proto().layer[0]

        # print(type(p1), dir(p1), dir(c))
        priorbox_layers[name].CopyFrom(L.PriorBox(min_size=min_size, clip=clip, variance=prior_variance, offset=offset).to_proto().layer[0])
        priorbox_layers[name].name = name
        priorbox_layers[name].top[0] = name
        #print(type(priorbox_layers[name]), dir(priorbox_layers[name].prior_box_param.max_size))
        priorbox_layers[name].bottom.append(from_layer)
        priorbox_layers[name].bottom.append(data_layer)

        if max_size: 
            priorbox_layers[name].prior_box_param.max_size.extend(max_size)
        if aspect_ratio:
            priorbox_layers[name].prior_box_param.aspect_ratio.extend(aspect_ratio)
            #priorbox_layers[name].prior_box_param.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if not flip:    #default is True
            priorbox_layers[name].prior_box_param.flip = flip
        if step:
            priorbox_layers[name].prior_box_param.step = step

    # Concatenate priorbox, loc, and conf layers.
    name = "mbox_priorbox"
    cat_mbox_layer = net.layer.add()
    cat_mbox_layer.CopyFrom(L.Concat(axis=2).to_proto().layer[0])
    cat_mbox_layer.name = name
    cat_mbox_layer.top[0] = name
    for bt in priorbox_layers.keys():
        cat_mbox_layer.bottom.append(bt)

def update_sample_channel(net, sample_channel_layers, prev_channels, sample_channel_index_file):
    if sample_channel_layers == None or os.path.isfile(sample_channel_index_file): return
    indexlists = open(sample_channel_index_file).readlines()
    layer_names = [l.name for l in net.layer]
    for idx, (layer, depth)  in enumerate(zip(sample_channel_layers, prev_channels)):
        name = layer + '_slice'
        layer_idx = layer_names.index(name)
        l = net.layer[layer_idx]
        slice_point = []
        cat_bottom = []
        cat_index = indexlists[idx].strip().split(' ')
        cat_index = [ int(float(i)) for i in cat_index ]
        for i in range(depth):
            if i == 0:
                l.top[i] = name + '_{}'.format(i)
            elif i > 0:
                l.top.append(name + '_{}'.format(i))
                slice_point.append(i)
            if i in cat_index: cat_bottom.append(name + '_{}'.format(i))

        l.slice_param.slice_point[0] = slice_point[0]
        l.slice_param.slice_point.extend(slice_point[1:])
        
        name = layer + '_cat'
        layer_idx = layer_names.index(name)
        l = net.layer[layer_idx]
        l.bottom[0] = cat_bottom[0]
        l.bottom.extend(cat_bottom[1:])

        pass
##########################################
conf_name = "Cat_5" #conf concat layer name
loc_name = "Cat_4"  #loc concat layer name
mbox_source_layers = ['Add1_1', 'ConvNd_23','Add1_2', 'Add1_3','ConvNd_46', 'ConvNd_49', 'ConvNd_52']
min_sizes = [[32], [52,65], [84,117,162], [233], [318],[421],[518]]   #in pixels
max_sizes = [] #[[12,13,14], [1,2,3]]

steps = [] #[8, 16]
aspect_ratios = [[2.0, 0.23], [0.25, 2.0,0.5], [0.4, 2.0], [2.0,0.6], [2.0,0.6], [1.5, 0.75],[2.0,0.5]]
# L2 normalize
normalizations = [-1, -1, -1, -1, -1, -1, -1]
# variance used to encode/decode prior bboxes.
prior_variance = [0.1, 0.1, 0.2, 0.2]
flip = False
clip = False
num_classes = 5 #add background
share_location = True
fn = './tmp_wd_2.prototxt' #caffe prototxt file path

####special for sample_channel layer, u can ignore if not use sample_channel
sample_channel_layers = ['IndexSelect_1', 'IndexSelect_2', 'IndexSelect_3', 'IndexSelect_4', 'IndexSelect_5', 'IndexSelect_6']
prev_channels = [128, 192, 256, 256, 256, 256]
sample_channel_index_file = 'index_right111.txt'

###################################
net = caffe_pb2.NetParameter()
with open(fn) as f:
    s = f.read()
    txtf.Merge(s, net)

#update_sample_channel(net, sample_channel_layers, prev_channels, sample_channel_index_file)

CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=False, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=1)

conf_name_std = 'mbox_conf'
reshape_name = "{}_reshape".format(conf_name_std)
tmp_layer = net.layer.add()
tmp_layer.CopyFrom(L.Reshape(shape=dict(dim=[0, -1, num_classes])).to_proto().layer[0])
tmp_layer.name = reshape_name
tmp_layer.top[0] = reshape_name
tmp_layer.bottom.append(conf_name)

softmax_name = "{}_softmax".format(conf_name_std)
tmp_layer = net.layer.add()
tmp_layer.CopyFrom(L.Softmax(axis=2).to_proto().layer[0])
tmp_layer.name = softmax_name
tmp_layer.top[0] = softmax_name
tmp_layer.bottom.append(reshape_name)

flatten_name = "{}_flatten".format(conf_name_std)
tmp_layer = net.layer.add()
#tmp_layer.CopyFrom(L.Flatten(axis=1).to_proto().layer[0])
tmp_layer.CopyFrom(L.Reshape(shape=dict(dim=[0, -1, 1, 1])).to_proto().layer[0])
tmp_layer.name = flatten_name
tmp_layer.top[0] = flatten_name
#print(type(priorbox_layers[name]), dir(priorbox_layers[name].prior_box_param.max_size))
tmp_layer.bottom.append(softmax_name)

det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': 0,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 200},
    'keep_top_k': 100,
    'confidence_threshold': 0.01,
    'code_type': P.PriorBox.CENTER_SIZE,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': 0,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    }

detection_out_name = "detection_out"
tmp_layer = net.layer.add()
tmp_layer.CopyFrom(L.DetectionOutput(detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST'))).to_proto().layer[0])
tmp_layer.name = detection_out_name
tmp_layer.top[0] = detection_out_name
tmp_layer.bottom.append(loc_name)
tmp_layer.bottom.append(flatten_name)
tmp_layer.bottom.append("mbox_priorbox")


#print(str(net))

# net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
#     detection_evaluate_param=det_eval_param,
#     include=dict(phase=caffe_pb2.Phase.Value('TEST')))

outFn = './newNet.prototxt'
with open(outFn, 'w') as f:
    f.write(str(net))
