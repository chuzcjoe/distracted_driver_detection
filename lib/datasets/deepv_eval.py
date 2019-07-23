import numpy as np

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:   #############invalid value encountered in greater_equal
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(gt_class_recs, pred_class_recs, npos, ovthresh=0.5, use_07_metric=True):
    image_ids = []
    confidence = []
    BB = []
    
    for im_idx in range(len(pred_class_recs)):
        dets = pred_class_recs[im_idx]
        if dets == []: continue
        for k in range(dets.shape[0]):
            image_ids.append(im_idx)
            confidence.append(dets[k, -1])
            BB.append([dets[k, 0], dets[k, 1],
                        dets[k, 2], dets[k, 3]])# don't +1 maybe result is influenced by test gt distriute
        # format: box_coord, conf
    if len(image_ids) == 0:
        return -1, -1, -1
    
    confidence = np.round_(np.array(confidence), 3)    #conf .3f
    BB = np.round_(np.array(BB), 1)  #bbox  .1f

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]

    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in gt_class_recs.keys():    #hasn't gts
            fp[d] = 1.
            continue
        R = gt_class_recs[image_ids[d]]    #gt with this classname in this image
        bb = BB[d, :].astype(float)   #predict bbox
        ovmax = -np.inf #init min_val
        BBGT = R['bbox'].astype(float)  #gt box
        
        if BBGT.size > 0:   #if exist bbox
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih    #a^b
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                    (BBGT[:, 2] - BBGT[:, 0]) *
                    (BBGT[:, 3] - BBGT[:, 1]) - inters)  #a v b
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:    #difficult bbox not in calc
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)  #cumulative sum
    tp = np.cumsum(tp)
    rec = tp / float(npos) #+1e-10 can not any use #recall             #invalid value encountered in true_divide
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)   #precision
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def evaluate_detections_deepv(gt_recs, all_boxes, num_class, test_size):
    aps = []
    for cls_idx in range(1, num_class):   #skip background
        gt_class_recs = {}
        npos = 0    # indifficult num_obj in a img
        for img_ind in range(test_size):
            if len(gt_recs[img_ind]) == 0: continue   #hasn't any gts
            
            R = [obj for obj in gt_recs[img_ind] if obj['cls_id'] == cls_idx - 1]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)   #-1 is True
            det = [False] * len(R)  #detect flag,0:undetected 1:detected
            npos = npos + sum(~difficult)   #only think difficult=0
            gt_class_recs[img_ind] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}
                                    
        rec, prec, ap = voc_eval(gt_class_recs, all_boxes[cls_idx], npos, 0.5, True)
        aps += [ap]
        res = aps
        mAP = np.mean(aps)
    return res, mAP