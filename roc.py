# -*- coding: UTF-8 -*-
import copy
import numpy as np
import os, re
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shutil
import sys
import argparse
import xml.dom.minidom
import xmltodict
import cv2
#import matplotlib
#matplotlib.use('Agg')

prob_thres = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 0.999]
prob_thres = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 0.922, 0.94, 0.96, 0.980, 0.985, 0.990, 0.995, 0.999]

def iou(b1, b2):#[xmin, ymin, xmax, ymax]
    iou_val = 0.0

    b1_w = b1[2] - b1[0]
    b1_h = b1[3] - b1[1]
    b2_w = b2[2] - b2[0]
    b2_h = b2[3] - b2[1]

    x1 = np.max([b1[0], b2[0]])
    y1 = np.max([b1[1], b2[1]])
    x2 = np.min([b1[2], b2[2]])
    y2 = np.min([b1[3], b2[3]])
    w = np.max([0, x2 - x1])
    h = np.max([0, y2 - y1])
    if w != 0 and h != 0:
        iou_val = float(w * h) / (b1_w * b1_h + b2_w * b2_h - w * h)

    return iou_val

def precision_recal(pred_origin, gt_origin, thres):

    p_num = len(pred_origin)  #pred total num
    r_num = len(gt_origin)    #gt total num
    p_cnt = 0
    r_cnt = 0
    if p_num == 0 or r_num == 0: return p_cnt, r_cnt, p_num, r_num
    gt = np.copy(gt_origin)
    pred = np.copy(pred_origin)

    flag = 0
    for b1 in pred:
        for idx, b2 in enumerate(gt):
            if iou(b1, b2) > thres:
                p_cnt += 1
                gt = np.delete(gt, [idx], axis=0)
                flag = 1
                break

    gt = np.copy(gt_origin)
    pred = np.copy(pred_origin)
    for b1 in gt:
        for idx, b2 in enumerate(pred):
            if iou(b1, b2) > thres:
                r_cnt += 1
                pred = np.delete(pred, [idx], axis=0)
                break

    return p_cnt, r_cnt, p_num, r_num


def get_gt(file_path, anno_key, resized_height = 0):
    gt = []
    #print file_path
    with open(file_path, 'r') as f:
        d = xmltodict.parse(f.read())
        anno = d['annotation']
        # folder = anno['folder']
        filename = anno['filename']
        width = anno['size']['width']
        height = anno['size']['height']
        depth = anno['size']['depth']

        if not 'object' in anno.keys():
            return np.array(gt)

        objs = anno['object']
        if not isinstance(objs, list):   #if len(objs) is one, objs will not be list
            objs = [objs]
        for obj in objs:
            name = obj['name'].lower().strip()
            if str(name) != anno_key: continue

            xmin = obj['bndbox']['xmin']
            ymin = obj['bndbox']['ymin']
            xmax = obj['bndbox']['xmax']
            ymax = obj['bndbox']['ymax']
            gt.append([int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))])

    return np.array(gt)

def get_pred(pred_dict, key_name):
    probs = []
    bboxs = []
    if not key_name in pred_dict.keys():
        return np.array(bboxs), np.array(probs)

    pred_lists = pred_dict[key_name]
    probs = []
    bboxs = []
    for item in pred_lists:
        score = item[0]
        pred = item[1]

        probs.append(score)
        bboxs.append(pred)

    return np.array(bboxs), np.array(probs)

def roc(test_gt_list, pred_dict, anno_key, iou_thres, save_dir):
    precisions = np.zeros((len(prob_thres), ), dtype=np.int32)
    recalls = np.zeros((len(prob_thres), ), dtype=np.int32)
    phits = np.zeros((len(prob_thres), ), dtype=np.int32)
    rhits = np.zeros((len(prob_thres), ), dtype=np.int32)
    
    error_list = []
    for idx, line in enumerate(test_gt_list):
        key_name, gt_file= line[0], line[1] #img_name, xml_file
        #print gt_file
        gt = get_gt(gt_file, anno_key)
        #print gt[0]
        #pred, probs = get_pred_txt(pred_file, key_name)
        pred_box, confs = get_pred(pred_dict, key_name)
        #print(pred_box)
        for k in range(len(prob_thres)):
            # print k
            if len(pred_box) > 0:
                I = confs > prob_thres[k]
                #prob_k = confs[I]
                pred_k = pred_box[I, :]
            else:
                pred_k = [] #pred_ori, gt_ori, gt_side_ori, thres
            
            pm, rm, pn, rn = precision_recal(pred_k, gt, iou_thres)
            #print(pm, rm, pn, rn)
            # if prob_thres[k] == 0.6 and pm < pn:
            #     error_list.append(key_name+'\n')

            phits[k] += pm
            rhits[k] += rm
            precisions[k] += pn
            recalls[k] += rn
        
        #if idx == 3: break
        if idx % 1000 == 0:
            print("processed {:d}/{:d}".format(idx, len(test_gt_list)))
    
    # fw = open('error_list', 'w')
    # fw.writelines(error_list)
    # fw.close()

    mean_p = np.zeros((len(prob_thres), ), dtype=np.float32)
    mean_r = np.zeros((len(prob_thres), ), dtype=np.float32)
    for k in range(len(prob_thres)):
        if precisions[k] == 0:  #pred_num = 0
            mean_r[k] = 0.0
            continue
        if recalls[k] == 0: #gt_num = 0
            mean_r[k] = 1.0
            continue

        mean_p[k] = phits[k] * 1.0 / precisions[k]
        mean_r[k] = rhits[k] * 1.0 / recalls[k]

    return mean_p, mean_r

#gen img_xml_list and read pred results to dict
def get_img_xml_pred(data_path, test_file, pred_file):
    img_xml_list = []
    pred_dict = {}

    lines = open(test_file, 'r').readlines()
    for _, line in enumerate(lines):
        image_path, xml_path = line.strip().split()
        abs_xml_path = os.path.join(data_path, xml_path)
        img_xml_list.append([image_path, abs_xml_path])
    
    lines = open(pred_file, 'r').readlines()
    for line in lines:
        split_item = line.strip().split()
        image_path = split_item[0]
        #image key first occur
        if not image_path in pred_dict.keys():
            pred_dict[image_path] = list()
        score = float(split_item[1])
        box = list(map(float, split_item[2:6]))
        pred_dict[image_path].append([score, list(map(int, box))])
    
    return img_xml_list, pred_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='roc')
    parser.add_argument('--data_path', dest='data_path', help='gt path', type=str)
    parser.add_argument('--test_file', dest='test_file', help='gt file path', type=str)
    parser.add_argument('--pred_file', dest='pred_file', help='pred file path', type=str)

    parser.add_argument('--save_dir', dest='save_dir', help='save dir', type=str)
    parser.add_argument('--anno_key', dest='anno_key', help='dataset key, use it to get gt', type=str)
    parser.add_argument('--resized_height', dest='resized_height', default=None, help='img height when test',
                        type=float)
    parser.add_argument('--ious', dest='ious', default="(0.3,0.4)", help='ious to compute roc, should be tuple or list',
                        type=str)
    parser.add_argument('--show', dest='show', help='whether to show plot figure',
                        action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    test_gt_list, pred_dict = get_img_xml_pred(args.data_path, args.test_file, args.pred_file)  #name, gt, pred file

    iou_list = eval(args.ious)  #convert str into compute value(e.g. list)
    if not isinstance(iou_list, list) and not isinstance(iou_list, tuple):
        iou_list = (iou_list,)
    
    stats = []
    for iou_thres in iou_list:
        mean_p, mean_r = roc(test_gt_list, pred_dict, args.anno_key, iou_thres, args.save_dir)
        stats.append([mean_p, mean_r])

    #plt.figure(figsize=(8, 7))
    for mean_p, mean_r in stats:
        print("error1")
        plt.plot(mean_r,  mean_p, 'o-')
        print("error2")
        for idx, x, y in zip(range(len(mean_p)), mean_r, mean_p):
            print(idx, 'thres =', prob_thres[idx], 'R =',x, 'P =',y)
            if idx % 2 == 0:
                plt.text(x+0.02, y+0.02, '%d(%.3f)'%(idx, prob_thres[idx]), rotation=60, ha='center', va='bottom', fontsize=7)
    
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("recall")
    plt.ylabel("precision")
    #plt.title(get_title(args.save_dir))
    plt.title(args.anno_key)
    #if args.show:
    #plt.show()
    plt.savefig("water.png")

# def get_test_gt_pred(data_path, test_file, pred_file):
#     test_gt_list = []
#     lines = open(test_file, 'r').readlines()
#     for _, line in enumerate(lines):
#         split_t = line.strip().split()
#         if len(split_t) == 2:
#             name, gt_xml = line.strip().split()
#             gt_xml = '{}/{}'.format(data_path, gt_xml)
#         test_gt_list.append([name, gt_xml])
    
#     pred_dict = {}
#     lines = open(pred_file, 'r').readlines()
#     for line in lines:
#         list_ = line.strip().split()
#         name = list_[0]     #image id
#         if not name in pred_dict.keys():
#             pred_dict[name] = list()
#         score = float(list_[1])
#         box = list(map(float, list_[2:6]))
#         pred_dict[name].append([score, list(map(int, box))])
#     return test_gt_list, pred_dict
