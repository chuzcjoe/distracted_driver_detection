"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse
import os, os.path as osp
import time
import numpy as np
import cv2 
import torch
from torch.autograd import Variable

from lib.utils.config import cfg, merge_cfg_from_file
from lib.datasets import dataset_factory
from lib.models import model_factory
from lib.utils import eval_solver_factory
from lib.utils.utils import setup_cuda, setup_folder

from lib.layers import DetectOut
import xml.etree.ElementTree as ET

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save_list2txt(img_gt_list, file_path):
    if img_gt_list is None or len(img_gt_list) == 0: return
    fw = open(file_path, 'w')
    fw.writelines(img_gt_list)
    fw.close()

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--cfg_name', default='ssd_vgg16_voc_re', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--job_group', default='rfbnet', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--trained_model', default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./results/debug', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int,
                    help='cpu workers for datasets processing')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--devices', default="0", type=str,
                    help='GPU to use')
parser.add_argument('--net_gpus', default=[0,], type=list,
                    help='GPU to use for net forward')
parser.add_argument('--loss_gpu', default=0, type=list,
                    help='GPU to use for loss calculation')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
# parser.add_argument('--show_test_image', default=False, type=str2bool,
#                     help='Cleanup and remove results files following eval')
parser.add_argument('--save_log', default=False, type=str2bool,
                    help='save log or not')


parser.add_argument('--test_path', default=None, type=str,
                    help='test path')
parser.add_argument('--vis', default=False, type=str2bool,
                    help='vis')
parser.add_argument('--crop', default=False, type=str2bool,
                    help='save crop')
args = parser.parse_args()


#python test.py --cfg_name res10_face_t --job_group face --trained_model ./weights/face/res10_face_t/res10_face_t_20000dark86.9.pth --test_path ./test_imgs --vis 1

class_ind = ['bg','car','person','bicycle','tricycle']
color_ind = [(255,0,0), (0,255,0),(0,0,255),(255,0,0),(255,255,0)]

label_name = ['drink','phone','hand','face']

classes_name = ['part_cover','all_cover','lp', 'nolp']
def save_xml(max_conf_bbx, img_name):
    xmin, ymin, xmax, ymax, _s, label = max_conf_bbx
    new_xml_path = './Annotations/{}.xml'.format(img_name[:-4])

    tree = ET.parse('/home/maolei/data/coverLP_det/coverlp_det_20181208/Annotations/193957909_1.xml')
    target = tree.getroot()
    name = None
    for obj in target.iter('object'):
        name = obj.find('name').text.lower().strip()
    
        if 'part_cover' in name:
            new_name = 'gg'
            obj.find('name').text = new_name
            bbox = obj.find('bndbox')
            bbox.find('xmin').text = str(int(xmin))
            bbox.find('ymin').text = str(int(ymin))
            bbox.find('xmax').text = str(int(xmax))
            bbox.find('ymax').text = str(int(ymax))
        else:
            #print('gg', name, img_path)
            pass
    tree.write(new_xml_path)

if __name__ == '__main__':
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg, phase='eval')
    cfg.DATASET.NUM_EVAL_PICS = 0
    cfg.EVAL.ONLY_SAVE_RESULTS = True
    cfg.DATASET.EVAL_BATCH_SIZE = 8
    cfg.DATASET.NUM_WORKERS = 2
    # cfg.DATASET.VAL_DATASET_DIR = '/home/maolei/data/coverLP_det/'
    # cfg.DATASET.TEST_SETS = (('test_data', 'small_test.txt'), )

    if tb_writer is not None:
        tb_writer.cfg['show_test_image'] = args.save_log
    model_dir = args.trained_model

    np.set_printoptions(precision=3, suppress=True, edgeitems=4)
    #loader = dataset_factory(phase='eval', cfg=cfg)

    # load net
    net, priors, _ = model_factory(phase='eval', cfg=cfg)
    # net.load_state_dict(torch.load(model_dir))
    net.load_state_dict(torch.load(model_dir)['state_dict'])

    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)
    net.eval()
    detector = DetectOut(cfg)

    print('test_type:', cfg.DATASET.TEST_SETS, 'test_model:', args.trained_model,
          'device_id:', args.devices, 'test_dir:', args.test_path)

    
    #files = os.listdir(args.test_dir)
    file_list = args.test_path
    if file_list[-1] == '/': file_list = file_list[:-1]
    data_name = os.path.splitext(os.path.basename(file_list))[0]
    parent_dir = os.path.dirname(os.path.abspath(file_list))
    save_path = parent_dir + '/{}_results/'.format(data_name)
    print('save_path:', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frame = 0
    save_list = []

    fw_rs = open(save_path + '/test_tmp.txt', 'w')

    if os.path.isdir(file_list):
        data_root = file_list + '/'
        lines = os.listdir(file_list)
    else:
        data_root = parent_dir+ '/'
        lines = open(file_list).readlines()
    
    for idx, f in enumerate(lines):
        #if idx > 10: break
        f = f.strip().split()[0]
        img_name = osp.basename(f)
        if frame % 100 == 0: print("processing ", frame) 
        if f[-3:] not in ['jpg', 'png', 'bmp']:
            print(f, 'is not image')
            continue
        
        img_root = os.path.join(data_root, f)
        img = cv2.imread(img_root)
        if img is None:
           print(img_root)
           continue
        im_copy = img.copy()
        h,w,c = img.shape
        x = cv2.resize(img, (cfg.DATASET.IMAGE_SIZE[1], cfg.DATASET.IMAGE_SIZE[0])).astype(np.float32)
        x -= (104., 117., 123.)
        x = x[:, :, (2, 1, 0)]
        
        x = torch.from_numpy(x).permute(2,0,1)
        x = Variable(x.unsqueeze(0)).cuda()
        # net = net.cuda()
        loc, conf = net(x, phase='eval')
        detections = detector(loc, conf, priors).data
        cnt = 0

        #xmin, ymin, xmax, ymax, score, cls
        max_conf_bbx = [-1., -1., -1., -1., -1., -1.] #conf idx
        for j in range(1, detections.size(1)):
            #print(j)
            dets = detections[0, j, :]
            label = label_name[j-1]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            
            for t in range(scores.shape[0]):
                
                if(scores[t] > 0.5):
                    x1 = min(boxes[t][0], w)
                    x1 = round(max(x1, 0), 1)
                    x2 = min(boxes[t][2], w)
                    x2 = round(max(x2, 0), 1)
                    y1 = min(boxes[t][1], h)
                    y1 = round(max(y1, 0), 1)
                    y2 = min(boxes[t][3], h)
                    y2 = round(max(y2, 0), 1)

                    if max_conf_bbx[4] < scores[t]:
                        max_conf_bbx[0] = x1
                        max_conf_bbx[1] = y1
                        max_conf_bbx[2] = x2
                        max_conf_bbx[3] = y2
                        max_conf_bbx[4] = scores[t]
                        max_conf_bbx[5] = j - 1
                    
                    if args.vis:
                        if True or (x2 - x1 > 20 and y2 - y1 > 20):
                            fw_rs.write(' '.join([f, str(scores[t]), str(j), str(x1), str(y1), str(x2), str(y2)]) + '\n')
                            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color_ind[j],2)
                            cv2.putText(img,label+'_'+str(scores[t]),(int(x1),int(y1)),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
                    if args.crop:
                        scale = 0.5
                        if x2-x1>30 and y2-y1>30: 
                            fw_rs.write(' '.join([f, str(scores[t]), str(j), str(x1), str(y1), str(x2), str(y2)]) + '\n')
                            roih, roiw = y2-y1, x2-x1
                            xmin = max(0, x1 - roiw*scale)
                            ymin = max(0, y1 - roih*scale)
                            xmax = min(w, x2 + roiw*scale)
                            ymax = min(h, y2 + roih*scale)
                            roi = im_copy[int(ymin):int(ymax), int(xmin):int(xmax)]

                            name=os.path.join(save_path, str(cnt)+'_'+f)
                            cv2.imwrite(name, roi)
                            cnt+=1
            
        if args.vis:
            cv2.imwrite(os.path.join(save_path, f.split('/')[-1]), img)
        frame += 1
 
    # save_list2txt(save_list, '{}_result.txt'.format(data_name))
    fw_rs.close()
  

