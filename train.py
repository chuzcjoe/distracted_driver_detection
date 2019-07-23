import argparse
import os, time
import os.path as osp
from distutils.dir_util import copy_tree
from shutil import copyfile

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from lib.utils.config import cfg
from lib.datasets import dataset_factory
from lib.models import model_factory
from lib.utils import eval_solver_factory
from lib.layers.modules import DetectLoss, matching, DetectLossPost
from lib.utils.utils import Timer, create_if_not_exist, setup_folder

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--cfg_name', default='rfb_vgg16_voc_orig',
                    help='base name of config file')
parser.add_argument('--job_group', default='face', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--devices', default=None, type=str,
                    help='GPU to use for net forward')
parser.add_argument('--net_gpus', default=None, type=list,
                    help='GPU to use for net forward')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=1, type=int,
                    help='Resume training at this iter')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--loss_gpu', default=0, type=list,
                    help='GPU to use for loss calculation')
parser.add_argument('--tensorboard', default=True, type=bool,
                    help='Use tensorboard')
parser.add_argument('--loss_type', default='ssd_loss', type=str,
                    help='ssd_loss only now')
parser.add_argument('--save_log', default=True, type=str2bool,
                    help='save log or not')
args = parser.parse_args()

# @profile
def train():
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg)
    step_index = 0

    train_loader = dataset_factory(phase='train', cfg=cfg)
    val_loader = dataset_factory(phase='eval', cfg=cfg)
    eval_solver = eval_solver_factory(val_loader, cfg)

    ssd_net, priors, _ = model_factory(phase='train', cfg=cfg)
    net = ssd_net  # net is the parallel version of ssd_net
    print(net)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_iter = checkpoint['iteration'] + 1
        step_index = checkpoint['step_index']
        ssd_net.load_state_dict(checkpoint['state_dict'])
    elif cfg.MODEL.PRETRAIN_MODEL != '':
        # pretained weights
        pretrain_weights = torch.load(cfg.MODEL.PRETRAIN_MODEL)
        if 'reducedfc' not in cfg.MODEL.PRETRAIN_MODEL:
            ssd_net.apply(weights_init)
            try:
                ssd_net.load_state_dict(pretrain_weights['state_dict'], strict=False)
            except RuntimeError:  # another dataset
                entries = [i for i in pretrain_weights['state_dict'].keys() if i.startswith('conf')]
                for key in entries:
                    del pretrain_weights['state_dict'][key]
                ssd_net.load_state_dict(pretrain_weights['state_dict'], strict=False)
        else:
            print('Loading base network...')
            ssd_net.base.load_state_dict(pretrain_weights)

            # initialize newly added layers' weights with xavier method
            print('Initializing weights...')
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
    else:
        print('Initializing weights...')
        ssd_net.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)



    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR,
                          momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
                          weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net, device_ids=cfg.GENERAL.NET_CPUS)
        priors = Variable(priors.cuda(cfg.GENERAL.LOSS_GPU), requires_grad=False)
        net = net.cuda()
    else:
        priors = Variable(priors, requires_grad=False)
    
    ssd_net.priors = priors
    ssd_net.criterion = DetectLoss(cfg)
    criterion_post = DetectLossPost(cfg)
    net.train()
    
    print('Using the specified args: \n', args)

    epoch_size = len(train_loader.dataset) // cfg.DATASET.TRAIN_BATCH_SIZE
    num_epochs = (cfg.TRAIN.MAX_ITER + epoch_size - 1) // epoch_size
    iteration = args.start_iter
    start_epoch = int(iteration * 1.0 / epoch_size)
    # continue training at 8w, 12w...
    if step_index > 0:
        adjust_learning_rate(optimizer, cfg.TRAIN.OPTIMIZER.LR, cfg.TRAIN.LR_SCHEDULER.GAMMA, 100,
                             step_index, None, None)

    # timer
    t_ = {'network': Timer(), 'misc': Timer(), 'eval': Timer()}
    t_['misc'].tic()
    iteration = args.start_iter
    for epoch in range(start_epoch, num_epochs):
        for images, targets, _ in train_loader:
            if iteration in cfg.TRAIN.LR_SCHEDULER.STEPS or (iteration <= cfg.TRAIN.WARMUP_EPOCH * epoch_size):
                if iteration in cfg.TRAIN.LR_SCHEDULER.STEPS: step_index += 1
                adjust_learning_rate(optimizer, cfg.TRAIN.OPTIMIZER.LR, cfg.TRAIN.LR_SCHEDULER.GAMMA, epoch,
                                     step_index, iteration, epoch_size, cfg.TRAIN.WARMUP_EPOCH)

            # save model
            if iteration % cfg.TRAIN.SAVE_ITER == 0 and iteration != args.start_iter or \
                    iteration == cfg.TRAIN.MAX_ITER:
                print('Saving state, iter:', iteration)
                save_checkpoint({'iteration': iteration,
                                 'step_index': step_index,
                                 'state_dict': ssd_net.state_dict()},
                                snapshot_dir,
                                args.cfg_name + '_' + repr(iteration) + '.pth')
            # Eval
            if iteration % cfg.TRAIN.EVAL_ITER == 0 or iteration == cfg.TRAIN.MAX_ITER:
                t_['eval'].tic()
                net.eval()
                aps, mAPs = eval_solver.validate(net, priors, tb_writer=tb_writer)
                net.train()
                t_['eval'].toc()
                print('Iteration ' + str(iteration) + ' || mAP: %.3f' % mAPs[0] + ' ||eval_time: %.4f/%.4f' %
                      (t_['eval'].diff, t_['eval'].average_time))
                if tb_writer is not None:
                    if cfg.DATASET.NAME == 'VOC0712' or 'FACE':
                        tb_writer.writer.add_scalar('mAP/mAP@0.5', mAPs[0], iteration)
                    else:
                        tb_writer.writer.add_scalar('mAP/mAP@0.5', mAPs[0], iteration)
                        tb_writer.writer.add_scalar('mAP/mAP@0.95', mAPs[1], iteration)

                if iteration == cfg.TRAIN.MAX_ITER:
                    break

            if args.cuda:
                images = Variable(images.cuda(), requires_grad=False)
                targets = [Variable(ann.cuda(cfg.GENERAL.LOSS_GPU), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            # forward
            t_['network'].tic()

            match_result = matching(targets, priors,
                                     cfg.LOSS.OVERLAP_THRESHOLD, cfg.MODEL.VARIANCE,
                                     args.cuda, cfg.GENERAL.LOSS_GPU, cfg=cfg)
            net_outputs = net(images, match_result=match_result, tb_writer=tb_writer)
            loss, (loss_l, loss_c) = criterion_post(net_outputs)

            loss_str = ' || Loss: %.3f' % (loss.data[0]) + '|| conf_loss: %.3f' % (loss_c) + ' || loc_loss: %.3f ' % (loss_l)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_['network'].toc()
            t_['misc'].toc()
            # log
            if iteration % cfg.TRAIN.LOG_LOSS_ITER == 0:
                current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
                print('Iter ' + str(iteration) + loss_str,end=' ')
                print('Timer: %.3f(%.3f) %.3f(%.3f) sec.' % (t_['misc'].diff, t_['misc'].average_time, t_['network'].diff, t_['network'].average_time),
                    'lr: %.6f' % optimizer.param_groups[0]['lr'], ' sys_time:', current_date)

                if tb_writer is not None:
                    phase = 'train'
                    tb_writer.writer.add_scalar('{}/loc_loss'.format(phase), loss_l, iteration)
                    tb_writer.writer.add_scalar('{}/conf_loss'.format(phase), loss_c, iteration)
                    tb_writer.writer.add_scalar('{}/all_loss'.format(phase), loss.data[0], iteration)
                    tb_writer.writer.add_scalar('{}/time'.format(phase), t_['misc'].diff, iteration)

            iteration += 1
            t_['misc'].tic()
    # backup_jobs(cfg, cfg_path, log_dir)


def backup_jobs(cfg, cfg_path, log_dir):
    print('backing up cfg and log')
    out_dir = osp.join(cfg.GENERAL.HISTORY_ROOT, cfg.GENERAL.JOB_GROUP, args.cfg_name)
    if osp.exists(out_dir):
        out_name = args.cfg_name + '_n'
        print('\033[91m' + 'backup with new name {}'.format(out_name) + '\033[0m')
        out_dir = osp.join(cfg.GENERAL.HISTORY_ROOT, cfg.GENERAL.JOB_GROUP, out_name)
    create_if_not_exist(out_dir)
    cfg_name = args.cfg_name + '.yml'

    copyfile(cfg_path, osp.join(out_dir, cfg_name))
    copy_tree(log_dir, out_dir)


def save_checkpoint(state, path, name):
    path_name = os.path.join(path, name)
    torch.save(state, path_name)


def adjust_learning_rate(optimizer, lr_origin, gamma, epoch,
                         step_index, iteration, epoch_size, warmup_epoch):
    if epoch < warmup_epoch:
        lr = 1e-6 + (lr_origin-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = lr_origin * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    train()
