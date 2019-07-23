import torch
import os
import sys
import time
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from collections import OrderedDict

def save_checkpoint(state, path, name):
    path_name = os.path.join(path, name)
    torch.save(state, path_name)

def point_form(boxes):
    #print(boxes[-1])
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax
def print_net_info(model):
    model_dict = model.state_dict()
    print(type(model_dict))
    for k in model_dict:
        print('key:', k)

def savenumpy2txt(fn, ary):
    print ('savenumpy2txt', ary.shape)
    flat = ary.reshape(-1)
    datasize = ary.size
    f = open(fn, 'w')
    for i in range(int((datasize + 3) / 4)):
        maxj = min((datasize - i * 4), 4)
        for j in range(maxj):
            f.write(str(flat[i * 4 + j]) + " ")
        f.write("\n")
    f.close()

def base_transform(image, size, mean):
    x = cv2.resize(image, size).astype(np.float32)
    x -= np.array(mean, dtype=np.float32)
    x = x.astype(np.float32)
    return x

#add hook
def get_layer_featuremaps(model, input_img, key=None):
    def forward_hook(self, input, output):  #Softmax can't call hook
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.data.size(), output.data.norm()))
        #print(self.__class__.__name__)
        feature_maps.append([output.data.cpu().numpy()])   #h, w
        #savenumpy2txt('/mnt/sdc1/maolei/data/dump/{}.txt'.format(key), output.data.cpu().numpy())
    
    feature_maps = []
    model_features_dict = OrderedDict()
    layer_id = 0

    for name, sub_module in model.named_modules():   
        if not isinstance(sub_module, nn.ModuleList) and \
                not isinstance(sub_module, nn.Sequential) and \
                type(sub_module) in nn.__dict__.values() and \
                not isinstance(sub_module, nn.Softmax):
            layer = sub_module
            #print(layer)
            if key is None:
                hook = layer.register_forward_hook(forward_hook)
                model_features_dict[name] = hook
            elif key == name:
                hook = layer.register_forward_hook(forward_hook)
                model_features_dict[name] = hook
                break
    
    x = Variable(input_img.unsqueeze(0))
    x = x.cuda()
    model(x)

    for idx, key in enumerate(model_features_dict.keys()):
        model_features_dict[key].remove()
        model_features_dict[key] = feature_maps[idx]

    return model_features_dict

#get layers info
def get_layers_info(net, img, layer_name=None, save_path=None):
    model_features_dict = OrderedDict()
    for name, sub_module in net.named_modules():   
            if not isinstance(sub_module, nn.ModuleList) and \
                    not isinstance(sub_module, nn.Sequential) and \
                    type(sub_module) in nn.__dict__.values() and \
                    not isinstance(sub_module, nn.Softmax):
                layer = sub_module  #e.g. nn.Conv  nn.ReLU ...
                #print(layer)
                model_features_dict[name] = []  #e.g. base.0.conv.0

    parameters_dict={}
    for name, param in net.named_parameters():  #get parm weights
        parameters_dict[name] = str(param.size())   #base.0.conv.0.weight base.0.conv.0.bias ...

    learning_weights = ['weight', 'bias', 'running_mean', 'running_var']
    model_keys = net.state_dict().keys()

    for key in model_features_dict:
        learning_weight = []
        for weight_type in learning_weights:
            if key+'.'+weight_type in model_keys:
                learning_weight.append(weight_type)
                if key+'.'+weight_type in parameters_dict.keys():
                    learning_weight.append(parameters_dict[key+'.'+weight_type])
                
        feature_map = get_layer_featuremaps(net, img, key)
        if layer_name is None:
            print(key, learning_weight, feature_map[key][0].shape)
        elif (key == layer_name) and (save_path is not None):
            savenumpy2txt('{}/{}.txt'.format(save_path, key), feature_map[key][0])
            pass#print(feature_map[key][0][0])

def get_receptive_filed(module):
    params = []
    for name, sub_module in module.named_modules():
        if (isinstance(sub_module, torch.nn.Conv2d) or \
            isinstance(sub_module, torch.nn.MaxPool2d)) and \
                            name.find('downsample') == -1:  #Resnet resdual
            kernel_size = sub_module.kernel_size \
                if isinstance(sub_module.kernel_size, tuple) \
                else (sub_module.kernel_size, sub_module.kernel_size)
            dilation = sub_module.dilation \
                if isinstance(sub_module.dilation, tuple) \
                else (sub_module.dilation, sub_module.dilation)
            stride = sub_module.stride \
                if isinstance(sub_module.stride, tuple) \
                else (sub_module.stride, sub_module.stride)
            params.append(['{}({})'.format(name, sub_module.__class__.__name__),
                           kernel_size,
                           dilation, stride])

    for k in range(len(params)):
        rf = np.array((1, 1))
        for i in range(k + 1)[::-1]:
            if params[i][2] != (1, 1):
                effective_kernel = (
                    np.array(params[i][1]) - 1) * (np.array(params[i][2])) + 1
            else:
                effective_kernel = np.array(params[i][1])
            if i == k:
                params[k].append(tuple(effective_kernel))
            rf = (rf - 1) * (np.array(params[i][3])) + effective_kernel

        params[k].append(tuple(rf))
    for v in params:
        print('name: {}\t kernel: {}\t dilation: {}\t stride: {}\t effect kernel: {}\t effect rf: {}\t'.format(
            v[0], v[1], v[2], v[3], v[4], v[5]))

def get_flops(net, input_shape=(1, 3, 300, 300)):
    from lib.utils.convert2caffe.flops_benchmark import add_flops_counting_methods, start_flops_count
    input = torch.randn(input_shape)
    input = torch.autograd.Variable(input.cuda())

    net = add_flops_counting_methods(net)
    net = net.train()
    net.start_flops_count()

    _ = net(input)

    return net.compute_average_flops_cost()/1e9/2

#######################decode box
def get_encode_box(model, input_img, targets):
    criterion = MultiBoxLossSSD(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                False, use_gpu=True)
    x = Variable(input_img.unsqueeze(0))
    print('x', type(x.data))
    x = x.cuda()
    out = model(x)

    loss_l, loss_c = criterion(out, targets)    #######ssd loss        
    loss = loss_l + loss_c
    print(loss.data[0])

if __name__ == '__main__':
    dataset_mean = (104, 117, 123)

    cfg = googlenet_cls #resnet18_cls #ssd_lite_deepv_mobilenet_rpn48  #ssd_deepv_shufflenet_v2_fpn #
    if cfg['ssds_type'] is not None:
        #init priors
        priorbox = PriorBox(cfg)
        num_box = priorbox.num_priors

        net, layer_dimensions = creat_model('train', cfg, num_box)
        priors = priorbox.forward(layer_dimensions)

        net.priors = Variable(priors, volatile=True)
        net.init_weight()

        print('Creating network base_model: {}, ssds_type: {}'.format(cfg['base_model'], cfg['ssds_type']))
        print('feature maps:', layer_dimensions)
        print('feature_layer', cfg['feature_layer'])
    else:
        net, _ = creat_model('train', cfg)

    #print_net_info(net)
    #net.load_state_dict(torch.load('../coverLP2caffe/coverlp_90000.pth')['state_dict'])   #model is dict{}

    #pre_process img
    resize_w = cfg['image_size'][1]-16
    resize_h = cfg['image_size'][0]-16

    cvimg = np.ones((resize_h, resize_h, 3)).astype('uint8') #cv2.imread('/mnt/sdc1/maolei/data/dump/test1.png')
    img = base_transform(cvimg, (resize_w, resize_h), dataset_mean)
    #img = img[:, :, (2, 1, 0)]
    img = torch.from_numpy(img).permute(2, 0, 1)

    #save img
    #savenumpy2txt('/mnt/sdc1/maolei/data/dump/{}.txt'.format("data"), img.numpy())
    #save priors
    #priors_point = point_form(priors)
    #savenumpy2txt('/mnt/sdc1/maolei/data/dump/{}.txt'.format("priors"), priors_point.numpy())

    net = net.cuda()
    cudnn.benchmark = False
    net.eval()  #very important here, otherwise batchnorm running_mean, running_var will be incorrect

    #print layer info
    get_layers_info(net, img, layer_name=None, save_path='../coverLP2caffe/')   #'classfiers.0'

    #get flops
    input_shape = (1, 3, resize_h, resize_w)
    total_flops = get_flops(net, input_shape)
    # For default vgg16 model, this shoud output 31.386288 G FLOPS
    print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(total_flops))

    #TODO is no work for multi path network
    #print receptive filed
    #get_receptive_filed(net)

    #model convert
    input_var = Variable(torch.rand(1, 3, resize_h, resize_w)).cuda()
    input_shape = (1, 3, resize_h, resize_w)
    net_name = 'cwl_shufflenet_v2_050_10_4_3_2_250000'
    #convert2caffe(net, input_shape, net_name, save_path='../../', save_graph=False):
    
    #get_encode_box
    targets = np.array([[1, 2, 3, 4, 0],[3, 3, 10, 10, 0]])
    targets = torch.from_numpy(targets).float()
    targets = [Variable(targets.cuda(), volatile=True)]

    net.train()
    #get_encode_box(net, img, targets)