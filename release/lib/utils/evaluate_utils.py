import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable

from lib.datasets.deepv_eval import evaluate_detections_deepv
from lib.datasets.voc_eval import get_output_dir, evaluate_detections
from lib.layers import DetectOut
from lib.utils import visualize_utils
from lib.utils.utils import Timer, AverageMeter


class EvalBase(object):
    def __init__(self, data_loader, cfg):
        self.detector = DetectOut(cfg)
        self.data_loader = data_loader
        self.dataset = self.data_loader.dataset
        self.name = self.dataset.name
        self.cfg = cfg
        self.results = None  # dict for voc and list for coco
        self.gt_recs = {}
        self.image_sets = self.dataset.image_sets

        self.data_root = self.data_loader.dataset.data_root
        self.ind_to_class = self.data_loader.dataset.ind_to_class

        self.img_ids = self.data_loader.dataset.ids.copy()
        if len(self.img_ids[0]) == 2:
            self.img_ids = [i[0] for i in self.img_ids]

    def reset_results(self):
        raise NotImplementedError
    
    def parse_rec(self, targets): #all objs in a img
        objects = []
        for obj in targets:
            if int(obj[-1]) == -1:  #no target form [-1]*6
                #print('Warning: no gts')
                break
            obj_struct = {}
            obj_struct['difficult'] = 0 #in coco.py, all difficult=0
            obj_struct['cls_id'] = int(obj[-1])  #start from 0
            obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                int(obj[2]), int(obj[3])]
            objects.append(obj_struct)

        return objects

    def convert_ssd_result(self, det, img_idx):
        """
        :param det:
        :param img_idx:
        :return: [xmin, ymin, xmax, ymax, score, image, cls, (cocoid)]
        """
        raise NotImplementedError

    def post_proc(self, det, gts, img_idx, id):
        raise NotImplementedError

    def evaluate_stats(self, classes=None, tb_writer=None):
        return NotImplementedError

    # @profile
    def validate(self, net, priors=None, criterion=None, use_cuda=True, tb_writer=None):
        print('start evaluation')
        priors = priors.cuda(self.cfg.GENERAL.NET_CPUS[0])
        self.reset_results()
        img_idx = 0
        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['misc'].tic()
        for batch_idx, (images, targets, extra) in enumerate(self.data_loader):
            if batch_idx % 25 == 0:
                print('processed image', img_idx)
            if use_cuda:
                images = Variable(images.cuda(), volatile=True)
                extra = extra.cuda()
            else:
                images = Variable(images, volatile=True)

            _t['im_detect'].tic()
            loc, conf = net(images, phase='eval')
            # image, cls, #box, [score, xmin, ymin, xmax, ymax]
            detections = self.detector(loc, conf, priors)
            _t['im_detect'].toc(average=False)
            
            # print(images, 'ssssssssbbbbbbbbbbb')
            det = detections.data
            # print(det)
            h = extra[:, 0].unsqueeze(-1).unsqueeze(-1)
            w = extra[:, 1].unsqueeze(-1).unsqueeze(-1)
            det[:, :, :, 1] *= w  # xmin
            det[:, :, :, 3] *= w  # xmax
            det[:, :, :, 2] *= h  # ymin
            det[:, :, :, 4] *= h  # ymax
            det, id = self.convert_ssd_result(det, img_idx)
            # the format is now xmin, ymin, xmax, ymax, score, image, cls, (cocoid)
            if tb_writer is not None and tb_writer.cfg['show_test_image']:
                self.visualize_box(images, targets, h, w, det, img_idx, tb_writer)  #NOTE targets is changed
            img_idx = self.post_proc(det, targets, img_idx, id)
        _t['misc'].toc(average=False)
        #print in eval.py
        if self.cfg.EVAL.ONLY_SAVE_RESULTS:
            print('model infer time', _t['im_detect'].total_time, _t['misc'].total_time)
        
        return self.evaluate_stats(None, tb_writer)

    def write_voc_results_file(self, num_class):
        save_path = self.data_root + '/results'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print('Writing results file in {}'.format(save_path))
        for cls_ind, cls_name in enumerate(self.ind_to_class):
            filename = '{}/det_test_{}.txt'.format(save_path, cls_name)
            with open(filename, 'w') as f:
                if cls_ind + 1 >= num_class:
                    continue
                for im_ind, img_name in enumerate(self.img_ids):
                    if im_ind >= len(self.img_ids) - self.cfg.EVAL.PAD_NUM:
                        break
                    dets = self.results[cls_ind+1][im_ind]    #skip background
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(img_name), dets[k, -1],
                                    dets[k, 0], dets[k, 1],
                                    dets[k, 2], dets[k, 3]))

    def visualize_box(self, images, targets, h, w, det, img_idx, tb_writer):
        det_ = det.cpu().numpy()
        # det_ = det_[det_[:, 4] > 0.5]
        images = images.permute(0, 2, 3, 1) #N H W C
        images = images.data.cpu().numpy()
        #print(det_.shape, 'ssss')
        for idx in range(len(images)):
            img = images[idx].copy()
            img += np.array((104., 117., 123.), dtype=np.float32)

            w_r = 300  # resize to 1000, h
            h_r = 300
            boxes = None
            det__ = np.array([])
            if det_.size != 0:
                det__ = det_[det_[:, 5] == idx]
                w_ = w[idx, :].cpu().numpy()
                h_ = h[idx, :].cpu().numpy()
                # w_r = 1000  # resize to 1000, h
                # h_r = w_r / w_ * h_
                det__[:, 0:4:2] = det__[:, 0:4:2] / w_ * w_r
                det__[:, 1:4:2] = det__[:, 1:4:2] / h_ * h_r

                if tb_writer.cfg['phase'] == 'train':
                    w_ = 1.
                    h_ = 1.
                t = targets[idx].clone().numpy()  # ground truth
                t[:, 0:4:2] = t[:, 0:4:2] / w_ * w_r
                t[:, 1:4:2] = t[:, 1:4:2] / h_ * h_r
                t[:, 4] += 1  # TODO because of the traget transformer  #label

                boxes = {'gt': t, 'pred': det__}    #Shape (*, 5) (*, 7)
            tb_writer.cfg['steps'] = img_idx + idx
            if self.name == 'MS COCO':
                tb_writer.cfg['img_id'] = int(det__[0, 7]) if det__.size != 0 else 'no_detect'
            if self.name == 'VOC0712' or 'TEMP_LP':
                tb_writer.cfg['img_id'] = int(det__[0, 5]) if det__.size != 0 else 'no_detect'
            tb_writer.cfg['thresh'] = 0.3
            visualize_utils.vis_img_box(img, boxes, (h_r, w_r), tb_writer) #(h_r, w_r)

class EvalDEEPV(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalDEEPV, self).__init__(data_loader, cfg)
        if cfg.DATASET.NUM_EVAL_PICS > 0:
            raise Exception("not support voc")

    def reset_results(self):
        self.results = [[[] for _ in range(len(self.dataset) - self.cfg.EVAL.PAD_NUM)]
                        for _ in range(self.cfg.MODEL.NUM_CLASSES)]

    def convert_ssd_result(self, det, img_idx):
        # append image id and class to the detection results by manually broadcasting
        id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)
        det = torch.cat((det, id, cls), 3)
        #print('det', det.size())
        mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())
        det = torch.masked_select(det, mymask).view(-1, 7)
        #print('det', det.size())
        # xmin, ymin, xmax, ymax, score, image, cls
        if det.dim() != 0:
            det = det[:, [1, 2, 3, 4, 0, 5, 6]]
        return det, id

    def post_proc(self, det, gts, img_idx, id):
        det = det.cpu().numpy()
        # det_tensors.append(det)
        # print('sss id', id.shape)
        for b_idx in range(id.shape[0]):#batch_num
            #print('sss det', det.shape)
            if det.size != 0:
                det_ = det[det[:, 5] == b_idx]
                # print('sss det_', det_.shape, b_idx)
                for cls_idx in range(1, id.shape[1]):  # skip bg class
                    det__ = det_[det_[:, 6] == cls_idx]
                    # print('sss det__', det__.shape, cls_idx)
                    if det__.size == 0:
                        continue
                    if img_idx < len(self.dataset) - self.cfg.EVAL.PAD_NUM:
                        self.results[cls_idx][img_idx] = det__[:, 0:5].astype(np.float32, copy=False)
            if img_idx < len(self.dataset) - self.cfg.EVAL.PAD_NUM:
                self.gt_recs[img_idx] = self.parse_rec(gts[b_idx])
            img_idx += 1
        return img_idx

    def evaluate_stats(self, classes=None, tb_writer=None):
        #only save results
        if self.cfg.EVAL.ONLY_SAVE_RESULTS:
            print('Hint: Only save results in one line!!!')
        
        self.write_oneline(self.cfg.MODEL.NUM_CLASSES)
        self.write_voc_results_file(self.cfg.MODEL.NUM_CLASSES)
        print('Evaluating detections')
        res, mAP = evaluate_detections_deepv(self.gt_recs, self.results, self.cfg.MODEL.NUM_CLASSES, len(self.dataset) - self.cfg.EVAL.PAD_NUM)
        # if tb_writer is not None and tb_writer.cfg['show_pr_scalar']:
        #     visualize_utils.viz_pr_curve(res, tb_writer)
        return res, [mAP]

    def write_oneline(self, num_class):
        save_path = self.data_root + '/results'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print('Writing oneline results file in {}'.format(save_path))
        cls_score = [[-1,-1] for _ in range(len(self.img_ids))]
        cls_w = [1.] * num_class #[1,0.09,1,0.08] #
        filename = '{}/oneline_{}.txt'.format(save_path, 'results')
        with open(filename, 'w') as f:
            for im_ind, img_name in enumerate(self.img_ids):
                if im_ind >= len(self.img_ids) - self.cfg.EVAL.PAD_NUM:
                    break
                f.write('{:s}'.format(str(img_name)))
                for cls_ind, cls_name in enumerate(self.ind_to_class):
                    if cls_ind + 1 >= num_class:
                        continue
                    dets = self.results[cls_ind+1][im_ind]    #skip background
                    if dets == []:
                        continue
                    if dets[0, -1] * cls_w[cls_ind] > cls_score[im_ind][1]:
                        cls_score[im_ind][1] = dets[0, -1] * cls_w[cls_ind]
                        cls_score[im_ind][0] = cls_ind
                    
                    for k in range(dets.shape[0]):
                        box_line = '{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(cls_ind+1, dets[k, -1],
                                                                            dets[k, 0], dets[k, 1],
                                                                            dets[k, 2], dets[k, 3])
                        f.write(' {}'.format(box_line))
                f.write('\n')
        
        test_fn = os.path.splitext(os.path.basename(self.image_sets[0][1]))[0]
    

class EvalVOC(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalVOC, self).__init__(data_loader, cfg)
        self.test_set = self.image_sets[0][1]
        if cfg.DATASET.NUM_EVAL_PICS > 0:
            raise Exception("not support voc")
        print('eval img num', len(self.dataset))

    def reset_results(self):
        self.results = [[[] for _ in range(len(self.dataset))]
                        for _ in range(self.cfg.MODEL.NUM_CLASSES)]

    def convert_ssd_result(self, det, img_idx):
        # append image id and class to the detection results by manually broadcasting
        id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)
        det = torch.cat((det, id, cls), 3)
        #print('det', det.size())
        mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())
        det = torch.masked_select(det, mymask).view(-1, 7)
        #print('det', det.size())
        # xmin, ymin, xmax, ymax, score, image, cls
        if det.dim() != 0:
            det = det[:, [1, 2, 3, 4, 0, 5, 6]]
        return det, id

    def post_proc(self, det, gts, img_idx, id):
        det = det.cpu().numpy()
        for b_idx in range(id.shape[0]):#batch_num
            if det.size != 0:
                det_ = det[det[:, 5] == b_idx]
                for cls_idx in range(1, id.shape[1]):  # skip bg class
                    det__ = det_[det_[:, 6] == cls_idx]
                    if det__.size == 0:
                        continue
                    if img_idx < len(self.dataset) - self.cfg.EVAL.PAD_NUM:
                        self.results[cls_idx][img_idx] = det__[:, 0:5].astype(np.float32, copy=False)
            if img_idx < len(self.dataset) - self.cfg.EVAL.PAD_NUM:
                self.gt_recs[img_idx] = self.parse_rec(gts[b_idx])
            img_idx += 1
        return img_idx

    def evaluate_stats(self, classes=None, tb_writer=None):
        output_dir = get_output_dir('ssd300_120000', self.test_set)
        # det_file = os.path.join(output_dir, 'detections.pkl')
        # with open(det_file, 'wb') as f:
        #     pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)
        print('Evaluating detections')
        res, mAP = evaluate_detections(self.results, output_dir, self.data_loader.dataset, test_set=self.test_set)
        # if tb_writer is not None and tb_writer.cfg['show_pr_scalar']:
        #     visualize_utils.viz_pr_curve(res, tb_writer)
        return res, [mAP]


#TODO COCO Eval is not surport
class EvalCOCO(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalCOCO, self).__init__(data_loader, cfg)
        if cfg.DATASET.NUM_EVAL_PICS > 0:
            self.dataset.ids = self.dataset.ids[:cfg.DATASET.NUM_EVAL_PICS]
        print('eval img num', len(self.dataset))

    def reset_results(self):
        self.results = []

    def convert_ssd_result(self, det, gts, img_idx):
        # append image id and class to the detection results by manually broadcasting
        id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)

        coco_id = torch.Tensor(self.dataset.ids[img_idx: img_idx + det.shape[0]])
        coco_id = coco_id.unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        det = torch.cat((det, id, cls, coco_id), 3)

        mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())
        det = torch.masked_select(det, mymask).view(-1, 8)
        # xmin, ymin, xmax, ymax, score, image, cls, cocoid
        det = det[:, [1, 2, 3, 4, 0, 5, 6, 7]]
        return det, id

    # @profile
    def post_proc(self, det, img_idx, id):
        # x1, y1, x2, y2, score, image, cls, cocoid
        det[:, 2] -= det[:, 0]  # w
        det[:, 3] -= det[:, 1]  # h
        # cocoid, x1, y1, x2, y2, score, cls
        det = det[:, [7, 0, 1, 2, 3, 4, 6]]
        det_ = det.cpu().numpy()
        # det__ = det_[det_[:, 5] > 0.5]
        self.results.append(det_)
        img_idx += id.shape[0]
        return img_idx

    def evaluate_stats(self, classes=None, tb_writer=None):
        from pycocotools.cocoeval import COCOeval
        res = np.vstack(self.results)
        for r in res:
            r[6] = self.dataset.target_transform.inver_map[r[6]]
        coco = self.dataset.cocos[0]['coco']
        coco_pred = coco.loadRes(res)
        cocoEval = COCOeval(coco, coco_pred, 'bbox')
        cocoEval.params.imgIds = self.dataset.ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        res = cocoEval.eval
        ap05 = res['precision'][0, :, :, 0, 2]
        map05 = np.mean(ap05[ap05 > -1])
        ap95 = res['precision'][:, :, :, 0, 2]
        map95 = np.mean(ap95[ap95 > -1])
        """
        # show precision of each class
        s = cocoEval.eval['precision'][0]
        t = s[:, :, 0, 2]
        m = np.mean(t[t>-1])
        rc = []
        for i in range(t.shape[1]):
            r = t[:, i]
            rc.append(np.mean(r[r>-1]))
        print(rc)
        """
        return res, [map05, map95]
