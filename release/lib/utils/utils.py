import os
import time
from os import path as osp

import torch
from torch.backends import cudnn

from lib.utils.config import merge_cfg_from_file
from lib.utils.visualize_utils import TBWriter
import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def setup_cuda(cfg, use_cuda, cuda_devices, net_gpus=None, loss_gpu=None):
    # set GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GENERAL.CUDA_VISIBLE_DEVICES \
        if cuda_devices is None else cuda_devices   #TODO bug: "" convert to list

    if net_gpus is not None:
        cfg.GENERAL.NET_CPUS = net_gpus
    if loss_gpu is not None:
        cfg.GENERAL.LOSS_GPU = loss_gpu
    # for profile
    os.environ["CUDA_LAUNCH_BLOCKING"] = cfg.GENERAL.CUDA_LAUNCH_BLOCKING
    if torch.cuda.is_available():
        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.deterministic = True  #TODO maybe increase results
            # cudnn.benchmark = False
        else:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def create_if_not_exist(names, warn=True):
    if not isinstance(names, list):
        names = [names]
    for name in names:
        if not osp.exists(name):  # snapshot and save_model
            os.mkdir(name)
        elif warn:
            print('\033[91m' + 'folder already exist! ' + '\033[0m')
            time.sleep(10)


def setup_folder(args, cfg, phase='train'):
    # setup_folder config
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    cfg_path = args.cfg_name #osp.join(cfg.GENERAL.CFG_ROOT, args.job_group, args.cfg_name+'.yml')
    args.cfg_name = os.path.basename(args.cfg_name)[:-4]
    merge_cfg_from_file(cfg_path)
    cfg.GENERAL.JOB_GROUP = args.job_group
    # setup_folder weights folder
    snapshot_dir = osp.join(cfg.GENERAL.WEIGHTS_ROOT, cfg.GENERAL.JOB_GROUP, args.cfg_name)
    warn = False if cfg.GENERAL.JOB_GROUP == 'tests' or 'debug' else True
    if phase == 'train':
        create_if_not_exist([cfg.GENERAL.WEIGHTS_ROOT, cfg.GENERAL.HISTORY_ROOT], warn=warn)
        create_if_not_exist([osp.join(cfg.GENERAL.WEIGHTS_ROOT, cfg.GENERAL.JOB_GROUP),
                             osp.join(cfg.GENERAL.HISTORY_ROOT, cfg.GENERAL.JOB_GROUP)], warn=warn)
        create_if_not_exist(snapshot_dir, warn=warn)
        log_dir = osp.join(osp.join(cfg.LOG.ROOT_DIR, cfg.GENERAL.JOB_GROUP + '_' + args.cfg_name + '_' + time_stamp))

    elif phase == 'eval':
        log_dir = osp.join(osp.join(cfg.LOG.ROOT_DIR, cfg.GENERAL.JOB_GROUP + '_' + args.cfg_name))
    # setup_folder logger
    # TODO image logger another writer
    tb_writer = None
    if args.save_log:
        tb_writer = TBWriter(log_dir, {'phase': phase,
                                    'show_pr_curve': cfg.LOG.SHOW_PR_CURVE,
                                    'show_test_image': cfg.LOG.SHOW_TEST_IMAGE,
                                    'show_pr_scalar': cfg.LOG.SHOW_PR_SCALAR,
                                    'show_maxconf_box': cfg.LOG.SHOW_MAXCONF_BOX})
    setup_cuda(cfg, args.cuda, args.devices, args.net_gpus, args.loss_gpu)
    return tb_writer, cfg_path, snapshot_dir, log_dir
