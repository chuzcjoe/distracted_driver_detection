import os.path
# import cv2
# cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

# gets home dir cross platform
# TODO used in lib/datasets, delete it
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

VARIANCE = [0.1, 0.2]  # detection.py use it   multibox_loss_ssd.py

# SSD300 CONFIGS
ssd_voc_vgg = {
    # dataset configs
    'dataset_name': 'VOC',
    'num_classes': 21,
    # model type
    'base_model': 'vgg16',
    'ssds_type': 'SSD',
    'prior_type': 'PriorBoxSSD',
    # model params
    'image_size': [300, 300],  # [H, W]
    # 'min_dim': 300,  # resize
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    # solver configs
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,  # 120000,
}

ssd_coco_vgg = {
    'num_classes': 81,
    'base_model': 'vgg16',
    'ssds_type': 'SSD_COCO',
    'prior_type': 'PriorBoxSSD',
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    # 'mbox_source_layers': [21, 33, 36, 38, 40, 42], #conv4_3
    'image_size': [300, 300],  # [H, W]
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'dataset_name': 'COCO',
}

# FPN
fpn_voc_vgg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,  # 120000,
    'image_size': [300, 300],  # [H, W]
    'min_dim': 300,  # resize
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'dataset_name': 'VOC',
    'base_model': 'vgg16',
    'ssds_type': 'fpn',
}

# FSSD300 CONFIGS
fssd_voc_vgg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [[30], [60], 111, 162, 213, 264],
    'max_sizes': [[60], [111], 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'dataset_name': 'VOC',
    'base_model': 'vgg16',
    'ssds_type': 'fssd',
    'TEST_SETS': [('2007', 'test_full')],
}

fssd_coco_vgg = {
    'num_classes': 81,  # 201????
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'flip': True,
    'dataset_name': 'COCO',
    'base_model': 'vgg16',
    'ssds_type': 'fssd',
}
