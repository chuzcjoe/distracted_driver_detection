import torch
import cv2
import numpy as np
import math
from tensorboardX import SummaryWriter


class TBWriter(object):
    """class contains tensorboard writer and its config"""
    def __init__(self, log_dir=None, cfg=None):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.cfg = cfg


def draw_bbox(img, bbxs, color=(255, 255, 0), thick=2, cfg=None,):
    img = img.copy()
    bboxes = bbxs.copy()
    if 'thresh' in cfg:
        bboxes = bboxes[bboxes[:, 4] > cfg['thresh']]
    for bbx in bboxes:  # coordinates shift in the aug process
        if bbx[2] <= 1:
            bbx[0] *= img.shape[1]
            bbx[1] *= img.shape[0]
            bbx[2] *= img.shape[1]
            bbx[3] *= img.shape[0]
        # bbx = bbx.astype(int)
        if len(bbx) <= 5:  # ground truth
            cv2.rectangle(img, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), (0, 0, 255), thick)
            if len(bbx) == 5:
                cv2.putText(img=img, text='{:d}'.format(int(bbx[4])),
                            org=(int(bbx[0]), int(bbx[1] + 20)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8,
                            color=(0, 0, 255), thickness=thick)
        elif len(bbx) > 6:  # prediction
            cv2.rectangle(img, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), color, thick)
            cv2.putText(img=img, text='{:d}_{:.3f}'.format(int(bbx[6]), bbx[4]),
                        org=(int(bbx[0]), int(bbx[3])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=color, thickness=thick)
    return img


def vis_img_box(img, boxes, resize, tb_writer):
    colors = [(255, 255, 0), (0, 255, 0)]
    image = img.copy()
    if resize is not None:
        image = cv2.resize(image, (resize[1], resize[0]))
    if not isinstance(boxes, dict):
        boxes = {'box': boxes}
    for key, color in zip(boxes.keys(), colors):
        thick = 2 if 'thick' not in tb_writer.cfg else tb_writer.cfg['thick']
        image = draw_bbox(image, boxes[key], color, thick, tb_writer.cfg)

    cv2.imwrite('./cache/temp.jpg', image)  # TODO solve this bug
    image = cv2.imread('./cache/temp.jpg')
    image = image[..., ::-1]  # to rgb

    if 'aug_vis_list' in tb_writer.cfg:
        tb_writer.writer.add_image('augmentation/%s' % tb_writer.cfg['aug_name'],
                               image, tb_writer.cfg['steps'])
    else:
        tb_writer.writer.add_image('postprocess2/{}_{}'.format(
            tb_writer.cfg['steps'], tb_writer.cfg['img_id']), image, 0)


def viz_pr_curve(res, tb_writer):
    """
    :param res: tuple of (cls, ap, prec, rec)
    :param tb_writer: TBWriter
    """
    iter = tb_writer.cfg['iter'] if 'iter' in tb_writer.cfg else 0
    phase = tb_writer.cfg['phase'] if 'phase' in tb_writer.cfg else 'eval'
    for cls, ap, prec, rec in res:
        tb_writer.writer.add_scalar('{}/{}'.format(phase, cls), ap, iter)
        if tb_writer.cfg['show_pr_curve']:
            num_thresholds = min(500, len(prec))
            if num_thresholds != len(prec):
                gap = int(len(prec) / num_thresholds)
                prec = np.append(prec[::gap], prec[-1])
                rec = np.append(rec[::gap], rec[-1])
                num_thresholds = len(prec)
            prec.sort()
            rec[::-1].sort()
            tb_writer.writer.add_pr_curve_raw(
                tag=cls,
                true_positive_counts=-np.ones(num_thresholds),
                false_positive_counts=-np.ones(num_thresholds),
                true_negative_counts=-np.ones(num_thresholds),
                false_negative_counts=-np.ones(num_thresholds),
                precision=prec,
                recall=rec,
                global_step=iter,
                num_thresholds=num_thresholds
            )


def images_to_writer(writer, images, prefix='image', names='image', epoch=0):
    if isinstance(names, str):
        names = [names + '_{}'.format(i) for i in range(len(images))]

    for image, name in zip(images, names):
        writer.add_image('{}/{}'.format(prefix, name), image, epoch)


def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    # print(image.shape)
    channel = image.shape[0]
    image = torch.sum(image, dim=0)
    # print(image.shape)
    image = torch.div(image, channel)
    # print(image.shape)
    # assert False
    return image


def to_image_size(feature, target_img):
    height, width, _ = target_img.shape
    resized_feature = cv2.resize(feature, (width, height))
    return resized_feature


def features_to_grid(features):
    num, height, width, channel = (len(features), len(features[0]), len(features[0][0]), len(features[0][0]))
    rows = math.ceil(np.sqrt(num))
    output = np.zeros([rows * (height + 2), rows * (width + 2), 3], dtype=np.float32)

    for i, feature in enumerate(features):
        row = i % rows
        col = math.floor(i / rows)
        output[row * (2 + height) + 1:(row + 1) * (2 + height) - 1,
        col * (2 + width) + 1:(col + 1) * (2 + width) - 1] = feature

    return output


def viz_feature_maps(writer, feature_maps, module_name='base', epoch=0, prefix='module_feature_maps'):
    feature_map_visualization = []
    for i in feature_maps:
        i = i.squeeze(0)
        temp = to_grayscale(i)
        feature_map_visualization.append(temp.data.cpu().numpy())

    names, feature_map_heatmap = [], []
    for i, feature_map in enumerate(feature_map_visualization):
        feature_map = (feature_map * 255)
        heatmap = cv2.applyColorMap(feature_map.astype(np.uint8), cv2.COLORMAP_JET)
        feature_map_heatmap.append(heatmap[..., ::-1])
        names.append('{}.{}'.format(module_name, i))

    images_to_writer(writer, feature_map_heatmap, prefix, names, epoch)


def viz_grads(writer, model, feature_maps, target_image, target_mean, module_name='base', epoch=0,
              prefix='module_grads'):
    grads_visualization = []
    names = []
    for i, feature_map in enumerate(feature_maps):
        model.zero_grad()

        # print()
        feature_map.backward(torch.Tensor(np.ones(feature_map.size())), retain_graph=True)
        # print(target_image.grad)
        grads = target_image.grad.data.clamp(min=0).squeeze(0).permute(1, 2, 0)
        # print(grads)
        # assert False
        grads_visualization.append(grads.cpu().numpy() + target_mean)
        names.append('{}.{}'.format(module_name, i))

    images_to_writer(writer, grads_visualization, prefix, names, epoch)


def viz_module_feature_maps(writer, module, input_image, module_name='base', epoch=0, mode='one',
                            prefix='module_feature_maps'):
    output_image = input_image
    feature_maps = []

    for i, layer in enumerate(module):
        output_image = layer(output_image)
        feature_maps.append(output_image)

    if mode is 'grid':
        pass
    elif mode is 'one':
        viz_feature_maps(writer, feature_maps, module_name, epoch, prefix)

    return output_image


def viz_module_grads(writer, model, module, input_image, target_image, target_mean, module_name='base', epoch=0,
                     mode='one', prefix='module_grads'):
    output_image = input_image
    feature_maps = []

    for i, layer in enumerate(module):
        output_image = layer(output_image)
        feature_maps.append(output_image)

    if mode is 'grid':
        pass
    elif mode is 'one':
        viz_grads(writer, model, feature_maps, target_image, target_mean, module_name, epoch, prefix)

    return output_image


# def add_pr_curve_raw(writer, tag, precision, recall, epoch=0):
#     num_thresholds = len(precision)
#     writer.add_pr_curve_raw(
#         tag=tag,
#         true_positive_counts = -np.ones(num_thresholds),
#         false_positive_counts = -np.ones(num_thresholds),
#         true_negative_counts = -np.ones(num_thresholds),
#         false_negative_counts = -np.ones(num_thresholds),
#         precision = precision,
#         recall = recall,
#         global_step = epoch,
#         num_thresholds = num_thresholds
#     )


# def viz_pr_curve(writer, precision, recall, epoch=0):
#     for i, (_prec, _rec) in enumerate(zip(precision, recall)):
#         # _prec, _rec = prec, rec
#         num_thresholds = min(500, len(_prec))
#         if num_thresholds != len(_prec):
#             gap = int(len(_prec) / num_thresholds)
#             _prec = np.append(_prec[::gap], _prec[-1])
#             _rec  = np.append(_rec[::gap], _rec[-1])
#             num_thresholds = len(_prec)
#         # the pr_curve_raw_data_pb() needs the a ascending precisions array and a descending recalls array
#         _prec.sort()
#         _rec[::-1].sort()
#         # TODO: need to change i to the name of the class
#         # 0 is the background class as default
#         add_pr_curve_raw(
#             writer=writer, tag='pr_curve/class_{}'.format(i+1), precision = _prec, recall = _rec, epoch = epoch )


def viz_archor_strategy(writer, sizes, labels, epoch=0):
    ''' generate archor strategy for all classes
    '''

    # merge all datasets into one
    height, width, max_size, min_size, aspect_ratio, label = [list() for _ in range(6)]
    for _size, _label in zip(sizes[1:], labels[1:]):
        _height, _width, _max_size, _min_size, _aspect_ratio = [list() for _ in range(5)]
        for size in _size:
            _height += [size[0]]
            _width += [size[1]]
            _max_size += [max(size)]
            _min_size += [min(size)]
            _aspect_ratio += [size[0] / size[1] if size[0] < size[1] else size[1] / size[0]]
        height += _height
        width += _width
        max_size += _max_size
        min_size += _min_size
        aspect_ratio += _aspect_ratio
        label += _label

    height, width, max_size, min_size, aspect_ratio = \
        np.array(height), np.array(width), np.array(max_size), np.array(min_size), np.array(aspect_ratio)
    matched_height, matched_width, matched_max_size, matched_min_size, matched_aspect_ratio = \
        height[label], width[label], max_size[label], min_size[label], aspect_ratio[label]

    num_thresholds = 100
    # height, width, max_size, min_size, aspect_ratio = \
    height.sort(), width.sort(), max_size.sort(), min_size.sort(), aspect_ratio.sort()
    # matched_height, matched_width, matched_max_size, matched_min_size, matched_aspect_ratio = \
    matched_height.sort(), matched_width.sort(), matched_max_size.sort(), matched_min_size.sort(), matched_aspect_ratio.sort()

    x_axis = np.arange(num_thresholds)[::-1] / num_thresholds + 0.5 / num_thresholds

    # height 
    gt_y, _ = np.histogram(height, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip(gt_y[::-1] / len(height), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_gt', precision=gt_y, recall=x_axis, epoch=epoch)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_gt_normalized', precision=gt_y / max(gt_y), recall=x_axis,
        epoch=epoch)

    matched_y, _ = np.histogram(matched_height, bins=num_thresholds, range=(0.0, 1.0))
    matched_y = np.clip(matched_y[::-1] / len(height), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_matched', precision=matched_y / gt_y, recall=x_axis,
        epoch=epoch)

    # width
    gt_y, _ = np.histogram(width, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip(gt_y[::-1] / len(width), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_gt', precision=gt_y, recall=x_axis, epoch=epoch)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_gt_normalized', precision=gt_y / max(gt_y), recall=x_axis,
        epoch=epoch)

    matched_y, _ = np.histogram(matched_width, bins=num_thresholds, range=(0.0, 1.0))
    matched_y = np.clip(matched_y[::-1] / len(width), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_matched', precision=matched_y / gt_y, recall=x_axis,
        epoch=epoch)

    # max_size
    gt_y, _ = np.histogram(max_size, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip(gt_y[::-1] / len(max_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_gt', precision=gt_y, recall=x_axis, epoch=epoch)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_gt_normalized', precision=gt_y / max(gt_y),
        recall=x_axis, epoch=epoch)

    matched_y, _ = np.histogram(matched_max_size, bins=num_thresholds, range=(0.0, 1.0))
    matched_y = np.clip(matched_y[::-1] / len(max_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_matched', precision=matched_y / gt_y, recall=x_axis,
        epoch=epoch)

    # min_size
    gt_y, _ = np.histogram(min_size, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip(gt_y[::-1] / len(min_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_gt', precision=gt_y, recall=x_axis, epoch=epoch)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_gt_normalized', precision=gt_y / max(gt_y),
        recall=x_axis, epoch=epoch)

    matched_y, _ = np.histogram(matched_min_size, bins=num_thresholds, range=(0.0, 1.0))
    matched_y = np.clip(matched_y[::-1] / len(min_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_matched', precision=matched_y / gt_y, recall=x_axis,
        epoch=epoch)

    # aspect_ratio
    gt_y, _ = np.histogram(aspect_ratio, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip(gt_y[::-1] / len(aspect_ratio), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_gt', precision=gt_y, recall=x_axis, epoch=epoch)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_gt_normalized', precision=gt_y / max(gt_y),
        recall=x_axis, epoch=epoch)

    matched_y, _ = np.histogram(matched_aspect_ratio, bins=num_thresholds, range=(0.0, 1.0))
    matched_y = np.clip(matched_y[::-1] / len(aspect_ratio), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_matched', precision=matched_y / gt_y, recall=x_axis,
        epoch=epoch)

