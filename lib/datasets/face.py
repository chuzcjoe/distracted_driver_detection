"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import cv2
import torch.utils.data as data
import xml.etree.ElementTree as ET
from lib.datasets.config import *  # HOME, VARIANCE
from lib.datasets.det_dataset import DetDataset

# TODO move these global variable
# 20 classes altogether
FACE_CLASSES = (
    'face',)



class FACEAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, norm_box=True, class_to_ind=None, keep_difficult=False):
        # TODO use label map
        self.class_to_ind = class_to_ind or dict(zip(FACE_CLASSES, range(len(FACE_CLASSES))))
        self.keep_difficult = keep_difficult
        self.norm_box = norm_box

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = 0
            # if obj.find('difficult'):
                # difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) # TODO one-based annotation?
                if self.norm_box:  # norm box using image height and width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]  # TODO, from 0-19 ?!
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class FACEDetection(DetDataset):
    """VOC Detection Dataset Object

     input is image, target is annotation

     Arguments:
         root (string): filepath to VOCdevkit folder.
         image_sets (list): imageset to use (eg. 'train', 'val', 'test')
         transform (callable, optional): transformation to perform on the
             input image
         target_transform (callable, optional): transformation to perform on the
             target `annotation`
             (eg: take in caption string, return tensor of word indices)
         dataset_name (string, optional): which dataset to load
             (default: 'VOC2007')
    """
    
    def __init__(self, root,
                 image_sets, classes_name,
                 transform=None, target_transform=None,
                 dataset_name='FACE', norm_box=True):
        super(FACEDetection, self).__init__(root, image_sets, dataset_name, transform, target_transform)
        global FACE_CLASSES
        FACE_CLASSES = classes_name
        self.ind_to_class = FACE_CLASSES

        self.target_transform = FACEAnnotationTransform(norm_box=norm_box)

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._setup()

    def _setup(self):
        for (year, name) in self.image_sets:
            rootpath = osp.join(self.data_root, year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def _pre_process(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)  # Shape(H, W, C)
        extra = img.shape
        return img, target, extra


def test_loader():
    # TODO: a strange bug: datasets loader hangs in cpu mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specified GPUs range
    val_loader = dataset_factory(phase='eval', cfg=cfg)
    for i, (images, targets, extra) in enumerate(val_loader):
        print(i)
        # print(targets)


def test_vis():
    cfg_name = 'test_data_face'
    cfg_path = osp.join(cfg.GENERAL.CFG_ROOT, 'tests', cfg_name+'.yml')
    merge_cfg_from_file(cfg_path)
    val_loader = dataset_factory(phase='train', cfg=cfg)
    dataset = val_loader.dataset
    log_dir = osp.join(osp.join(cfg.LOG.ROOT_DIR, 'tests' + '_' + cfg_name))

    # tb_writer = TBWriter(log_dir, {'aug_vis_list': [3, 4, 5, 6, 8]})
    tb_writer = TBWriter(log_dir, {'aug_vis_list': [4, 5, 8]})
    # tb_writer = None
    for img_idx in range(len(dataset)):
        # if img_idx >= 100:
        #     break
        tb_writer.cfg['aug'] = True
        tb_writer.cfg['steps'] = img_idx
        tb_writer.cfg['img_id'] = img_idx
        tb_writer.cfg['thick'] = 1
        image, target, extra = dataset.pull_item(img_idx, tb_writer)
        print(image.shape)
        tb_writer.writer.file_writer.flush()


if __name__ == '__main__':
    from lib.utils.config import cfg, merge_cfg_from_file
    from lib.utils.visualize_utils import TBWriter
    from lib.datasets import dataset_factory

    # test_loader()
    test_vis()
