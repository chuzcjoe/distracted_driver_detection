"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import torch
from torch.autograd import Variable

from lib.utils.config import cfg, merge_cfg_from_file
from lib.datasets import dataset_factory
from lib.models import model_factory
from lib.utils import eval_solver_factory
from lib.utils.utils import setup_cuda, setup_folder


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--cfg_name', default='ssd_vgg16_voc_re', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--job_group', default='face', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--trained_model', default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./results/debug', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int,
                    help='cpu workers for datasets processing')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--devices', default='0', type=str,
                    help='GPU to use')
parser.add_argument('--net_gpus', default=[0], type=list,
                    help='GPU to use for net forward')
parser.add_argument('--loss_gpu', default=0, type=list,
                    help='GPU to use for loss calculation')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

parser.add_argument('--save_log', default=False, type=str2bool,
                    help='save log or not')

parser.add_argument('--data_path', default='None', type=str,
                    help='data path')
parser.add_argument('--test_file', default='None', type=str,
                    help='test_file')


args = parser.parse_args()

def pytorch2caffe(net, use_cuda, save_graph):
    from lib.utils.convert2caffe.ConvertModel import ConvertModel_caffe
    save_path = cfg.DATASET.DATASET_DIR+'/'
    if 'CLS' in cfg.MODEL.TYPE:
        input_shape = (1, 3) + cfg.DATASET.TARGET_SIZE
    else:
        input_shape = (1, 3) + cfg.DATASET.IMAGE_SIZE
    model_name = osp.basename(args.trained_model)[:-4] #str(net.__class__.__name__) + '_caffe'
    text_net, binary_weights = ConvertModel_caffe(net, input_shape, softmax=False, use_cuda=use_cuda, save_graph=save_graph)
    import google.protobuf.text_format
    with open(save_path+model_name + '.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(text_net))
    with open(save_path+model_name + '.caffemodel', 'w') as f:
        f.write(binary_weights.SerializeToString())
    exit(0)

def print_flops(net):
    input_shape = (1, 3) + cfg.DATASET.IMAGE_SIZE
    from lib.utils.net_info import get_flops
    total_flops = get_flops(net, input_shape)
    print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(total_flops))
    #exit(0)

if __name__ == '__main__':
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg, phase='eval')
    merge_cfg_from_file(cfg_path)
    cfg.DATASET.NUM_EVAL_PICS = 0

    # args.trained_model = './results/vgg16_ssd_coco_24.4.pth'
    # args.trained_model = './results/ssd300_mAP_77.43_v2.pth'
    # args.trained_model = './res10_face_t_1500089.4586.8addcaradddark.pth'

    setup_cuda(cfg, args.cuda, args.devices)

    np.set_printoptions(precision=3, suppress=True, edgeitems=4)

    # load net
    net, priors, _ = model_factory(phase='eval', cfg=cfg)
    ###get flops
    print_flops(net)

    # net.load_state_dict(torch.load(model_dir))
    net.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc:storage)['state_dict'])

    ### pytorch2caffe
    convert2caffe = False
    if convert2caffe:
        #NOTE convert2caffe not need cuda
        use_cuda = False
        save_graph = True
        pytorch2caffe(net, use_cuda, save_graph)
        exit(0)
    
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)
    net.eval()

    print('test_type:', cfg.DATASET.TEST_SETS, 'test_model:', args.trained_model,
          'device_id:', cfg.GENERAL.CUDA_VISIBLE_DEVICES)
    
    loader = dataset_factory(phase='eval', cfg=cfg)
    eval_solver = eval_solver_factory(loader, cfg)
    res, mAPs = eval_solver.validate(net, priors, tb_writer=tb_writer)
    print('final mAP', mAPs)
