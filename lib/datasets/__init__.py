import torch.utils.data as data
from .det_dataset import detection_collate
from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES
from .coco import COCODetection, COCOAnnotationTransform, get_label_map
from .face import FACEDetection, FACEAnnotationTransform, FACE_CLASSES
from .temp_lp import TEMP_LPDetection, TEMP_LPAnnotationTransform, LMDBDetDataset

from .config import *
from lib.utils.augmentations import SSDAugmentation
from lib.utils.rfn_augment import RFBAugmentation

dataset_map = {'FACE': FACEDetection,
               'TEMP_LP': TEMP_LPDetection,
               'LMDBDet': LMDBDetDataset,
               'VOC0712': VOCDetection,
               'COCO2014': COCODetection}

augmentation_map = {
    'SSD': SSDAugmentation,
    'RFB': RFBAugmentation
}


def dataset_factory(phase, cfg):
    dataset_type = dataset_map[cfg.DATASET.NAME]
    aug_type = augmentation_map[cfg.AUGMENTATION.NAME]
    if phase == 'train':
        dataset = dataset_type(cfg.DATASET.DATASET_DIR, cfg.DATASET.TRAIN_SETS, cfg.DATASET.CLASSES_NAME,
                            aug_type(cfg, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS))
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                                    num_workers=cfg.DATASET.NUM_WORKERS,
                                    shuffle=True, collate_fn=detection_collate,
                                    pin_memory=True, drop_last=True)
        
    elif phase == 'eval':
        dataset = dataset_type(cfg.DATASET.VAL_DATASET_DIR, cfg.DATASET.TEST_SETS, cfg.DATASET.CLASSES_NAME,
                            aug_type(cfg, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, use_base=True),
                            norm_box=False)
        #NOTE support multi_gpus eval. Make eval img num % num_gpus == 0
        if len(dataset) % len(cfg.GENERAL.NET_CPUS) != 0:
            cfg.EVAL.PAD_NUM = (len(cfg.GENERAL.NET_CPUS) - len(dataset) % len(cfg.GENERAL.NET_CPUS))
        for i in range(cfg.EVAL.PAD_NUM):
            dataset.ids.append(dataset.ids[-1])
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.EVAL_BATCH_SIZE,
                                    num_workers=cfg.DATASET.NUM_WORKERS, shuffle=False,
                                    collate_fn=detection_collate, pin_memory=True)
    else:
        raise Exception("unsupported phase type")
    
    return data_loader
