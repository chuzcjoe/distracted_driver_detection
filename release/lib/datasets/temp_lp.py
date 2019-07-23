
import os.path as osp
import cv2
import torch.utils.data as data
import xml.etree.ElementTree as ET
from lib.datasets.det_dataset import DetDataset
import numpy as np
from six.moves import urllib
import pickle
#import lmdb

TEMP_LP_CLASSES = ('part_cover','all_cover','lp', 'nolp', 'dirty_cover','other_cover','blur','light')

class LMDBDetDataset(DetDataset):
    def __init__(self, root,
                 image_sets, classes_name,
                 transform=None, target_transform=None,
                 dataset_name='TEMP_LP', norm_box=True):
        super(LMDBDetDataset, self).__init__(root, image_sets, dataset_name, transform, target_transform)
        global TEMP_LP_CLASSES
        TEMP_LP_CLASSES = classes_name
        self.ind_to_class = TEMP_LP_CLASSES
        self.testpath = osp.join('%s', '%s', '%s')
        self.norm_box = norm_box

        self.keys = []
        self.env = []
        self.num_list = []
        self._setup()

    def _setup(self):
        for (data_name, file_name) in self.image_sets:
            test_file = self.testpath % (self.data_root, data_name, file_name)
            self.env.append(lmdb.open(test_file, subdir=osp.isdir(test_file),
                             readonly=True, lock=False, readahead=False, meminit=False))
            with self.env[-1].begin(write=False) as txn:
                # length = pa.deserialize(txn.get(b'__len__'))
                self.keys += pickle.loads(txn.get(b'__keys__'))
            self.num_list.append(len(self.keys))
        
        #{'lmdb_idx':key_idx, 'img':str(b, encoding = "utf8"),} 
        self.ids = [(str(b, encoding = "utf8"), key_idx) for key_idx, b in enumerate(self.keys)]
    

    def _pre_process(self, index):
        info = self.ids[index]
        choose_key = self.keys[info[1]]
        byteflow = None
        for j in range(len(self.env)):
            if index < self.num_list[j]:
                with self.env[j].begin(write=False) as txn:
                    byteflow = txn.get(choose_key)
                break
        if byteflow is None: #for pad ids in test phase
            with self.env[-1].begin(write=False) as txn:
                byteflow = txn.get(choose_key)
        
        unpacked = pickle.loads(byteflow)
        # load image
        imgbuf = unpacked[0]
        image = np.asarray(bytearray(imgbuf), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        extra = img.shape
        # load label
        pts = unpacked[1]
        target = []
        div_t = np.array([extra[1], extra[0], extra[1], extra[0], 1]).astype(np.float)
        for idx, cur_pt in enumerate(pts):
            bndbox = cur_pt
            if self.norm_box:# scale height and width
                bndbox = np.array(cur_pt) / div_t
            target.append(bndbox)
        
        return img, target, extra


class TEMP_LPAnnotationTransform(object):
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
        self.class_to_ind = class_to_ind or dict(zip(TEMP_LP_CLASSES, range(len(TEMP_LP_CLASSES))))
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
            if (obj.find('difficult')):
                difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            if name not in ['poi_palm','poi_water','poi_phone']: continue
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            box_pixel = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))    #x,y start from 0
                box_pixel.append(cur_pt)
                
                if self.norm_box:# scale height and width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            
            if self.norm_box:   # True during training filter small box
                 if max(box_pixel[2]-box_pixel[0], box_pixel[3]-box_pixel[1]) < 0.01: continue
                 if (box_pixel[2]-box_pixel[0]) * (box_pixel[3]-box_pixel[1]) < 20: continue
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
         
            res += [bndbox]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class TEMP_LPDetection(DetDataset):
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
                 dataset_name='TEMP_LP', norm_box=True):
        super(TEMP_LPDetection, self).__init__(root, image_sets, dataset_name, transform, target_transform)
        global TEMP_LP_CLASSES
        TEMP_LP_CLASSES = classes_name
        self.ind_to_class = TEMP_LP_CLASSES

        self.target_transform = TEMP_LPAnnotationTransform(norm_box=norm_box)
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._setup()

    def _setup(self):
        for (year, name) in self.image_sets:
            rootpath = osp.join(self.data_root, year)
            for line in open(osp.join(rootpath, name)):
                if ';' not in line:
                    split_item = line.strip().split()
                else:
                    split_item = line.strip().split(';')
                if len(split_item) != 2:
                    img_path = split_item[0]
                    xml_path = None
                else:
                    img_path, xml_path = split_item
                    if '.xml' not in xml_path: xml_path = None
                self.ids.append((img_path, xml_path))
                

    def _pre_process(self, index):
        img_path, xml_path = self.ids[index]
        
        if xml_path is not None:
            target = ET.parse(self.data_root+xml_path).getroot()
        else:
            target = None
        
        if 'http' not in img_path:
            img = cv2.imread(osp.join(self.data_root, img_path))  # Shape(H, W, C)
        else:
            try:
                resp = urllib.request.urlopen(img_path)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except:
                img = np.zeros((50,50,3)).astype("uint8")
                print('error img', img_path)
                target = None
        if img is None:
            raise SystemExit('Unable read image:', '{}'.format(self.data_root+img_path))
        extra = img.shape
     
        return img, target, extra

if __name__ == "__main__":
    import sys, os
    HOME = os.path.expanduser("~")
    import copy
    import cv2
    import random
    import matplotlib.pyplot as plt

    dataset_root = os.path.join(HOME, "data/coverLP_det/")   #BaseTransform  #SSDAugmentation
    dataset = TEMP_LPDetection(dataset_root, (('coverlp_clean', 'coverlp_clean.lst'),('13province_lp_5k','train.txt'),
              ('coverlp_det_20181207','train.txt'), ('dirtylp_20190319','train.txt'),('hardsample_data','hardsample_190405.txt'),
              ('hardsample_data', 'othercover_190411.txt'),('hardsample_data', 'blur_190412.txt'),('hardsample_data', 'light_190412.txt'),
              ('hardsample_data', 'qjdisk4_18120405.txt'),('hardsample_data', 'qjdisk4_181205.txt')), TEMP_LP_CLASSES)

    # dataset = TEMP_LPDetection(dataset_root, (('hardsample_data','othercover_190411.txt'),), TEMP_LP_CLASSES)
    save_list = []
    random.shuffle(dataset.ids)
    for idx in range(len(dataset)):
        if idx > 6000: break
        im, gt, extra = dataset.ids[idx]
        # label = gt[0][-1]
        img_path = extra[-1]
        save_list.append(img_path+'\n')
        
        if idx % 3000 == 0: print(idx)
    f = open('all.txt','w')
    f.writelines(save_list)
    f.close()


