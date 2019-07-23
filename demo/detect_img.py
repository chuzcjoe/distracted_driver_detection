from __future__ import print_function
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from data.config import *
from models.model_build import create_model
from utils import *
from layers import *

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def save_img(image, gt, top_boxes, line_width = 2):

    image.save('tmp_src_{}.jpg'.format(image.size[0]))
    print(top_boxes)
    print('\n\n', gt)
    #width, height = im.size
    draw = ImageDraw.Draw(image)
    for i in range(top_boxes.shape[0]):
        xmin = int(round(top_boxes[i][0] * image.size[0]))    #image.size[0]
        ymin = int(round(top_boxes[i][1] * image.size[1]))
        xmax = int(round(top_boxes[i][2] * image.size[0]))
        ymax = int(round(top_boxes[i][3] * image.size[1]))
        # score = top_conf[i]
        # label = int(top_label_indices[i])
        # label_name = top_labels[i]
        # display_txt = '%s: %.2f'%(label_name, score)
        for j in range(line_width):
            draw.rectangle((xmin+j, ymin+j, xmax-j, ymax-j), outline='red')
        draw.text((xmin, ymin), 'test', fill='red')
    
    for i in range(gt.shape[0]):
        xmin = int(round(gt[i][0]))
        ymin = int(round(gt[i][1]))
        xmax = int(round(gt[i][2]))
        ymax = int(round(gt[i][3]))
        # score = top_conf[i]
        # label = int(top_label_indices[i])
        # label_name = top_labels[i]
        # display_txt = '%s: %.2f'%(label_name, score)
        for j in range(line_width):
            draw.rectangle((xmin+j, ymin+j, xmax-j, ymax-j), outline='blue')
        draw.text((xmin, ymin), 'gt', fill='blue')
    image.save('tmp_{}.jpg'.format(image.size[0]))

def parse_rec(targets, w): #all objs in a img
    scale = np.array([w, 0, w, 0])
    objects = targets
    for obj in targets: #[obj for obj in recs[imagename] if obj['name'] == classname]
        obj_2 = obj[:-1] + scale
        obj_2 = np.append(obj_2, obj[-1])
        objects = np.row_stack((objects, obj_2))

    return objects

dataset_mean = (104, 117, 123)
voc_root = VOC_ROOT
cfg = ssd_voc_vgg
trained_model = '../../../weights/ssd_voc_best_120016.pth'

if __name__ == '__main__':
    input_h = 300
    input_w = 300

    is_combine = False
    if is_combine:
        input_w *= 2
        cfg['image_size'] = [input_h, input_w]
    # load net
    net, layer_dimensions = create_model(phase='test', cfg=cfg, input_h = input_h, input_w = input_w)
    priorbox = PriorBox(cfg)
    priors = priorbox.forward(layer_dimensions) #<class 'torch.FloatTensor'>???????

    net.priors = Variable(priors, volatile=True)

    net.load_state_dict(torch.load(trained_model)['state_dict'])   #model is dict{}
    net = net.cuda()
    cudnn.benchmark = False

    net.eval()
    print('Finished loading model!')
    # load datasets
    dataset = VOCDetection(voc_root, [('2007', 'test')],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform(False))
    
    # evaluation
    num_images = len(dataset)
    test_file = open('{}/VOC2007/ImageSets/Main/test.txt'.format(voc_root), 'r')
    test_list = test_file.readlines()

    
    for i, img_id in enumerate(test_list):
        if i >= 1: break
        im, gt, h, w, _, _ = dataset.pull_item(i)

        if is_combine:
            im = im.numpy()
            
            combine_img = np.concatenate((im, im), axis = 2)
            print('img shape:', combine_img.shape, 'hw:', h, w)
            print('pixel:', im[0][0][0], combine_img[0][0][300])
            print('pixel1:', im[0][150][150], combine_img[0][150][450])
            im = torch.from_numpy(combine_img)

        x = Variable(im.unsqueeze(0))
        x = x.cuda()
        
        detections = net(x).data

        # skip j = 0, because it's the background class
        top_boxes=[]
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            scores = dets[:, 0].cpu().numpy()
            boxes = boxes.cpu().numpy()

            top_indices = [i for i, conf in enumerate(scores) if conf >= 0.5]
            #print('debug', top_indices)
            if len(top_indices) == 0:
                continue
            top_boxes += boxes[top_indices].tolist()
        
        cvimg = cv2.imread('{}/VOC2007/JPEGImages/{}.jpg'.format(voc_root, img_id.strip()))
        
        if is_combine:
            cvimg = np.concatenate((cvimg, cvimg), axis = 1)
            gt = parse_rec(gt, w)
        #nimg = np.array(cvimg.swapaxes(1,2).swapaxes(0,1), dtype=np.float32)    #(C, H, W)
        #combine_img = combine_img[:, :, (2, 1, 0)]
        combine_img = Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))

        
        #print('gt shape:', combine_img.shape, gt.shape)


        save_img(combine_img, gt, np.array(top_boxes))
        
