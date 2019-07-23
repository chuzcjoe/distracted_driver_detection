<<<<<<< HEAD
<<<<<<< HEAD
# Pytorch SSD Series
## Support Arc:
* SSD [SSD: Single Shot Multibox  Detector](https://arxiv.org/abs/1512.02325)
* FSSD [FSSD: Feature Fusion Single Shot Multibox Detector](https://arxiv.org/abs/1712.00960)
* RFB-SSD[Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767)
* RefineDet[Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf)

### VOC2007 Test
| System                 |  *mAP*   | **FPS** (Titan X Maxwell) | **Our mAP** |
| :--------------------- | :------: | :-----------------------: | :---------: |
| Faster R-CNN (VGG16)   |   73.2   |             7             |             |    
| YOLOv2 (Darknet-19)    |   78.6   |            40             |             |
| R-FCN (ResNet-101)     |   80.5   |             9             |             |
| SSD300* (VGG16)        |   77.2   |            46             |    77.6     |  77.5(locw1.5) 77.8（locw2）
| SSD512* (VGG16)        |   79.8   |            19             |             |
| RFBNet300 (VGG16)      | **80.5** |            83             |             |
| RFBNet512 (VGG16)      | **82.2** |            38             |             |
| FSSD300 (VGG)          |   78.8   |       120 (1080Ti)        |             |
| FPN300 (VGG)           |          |                           |    78.3     |

### COCO 
| System                       | *test-dev mAP* | **Time** (Titan X Maxwell) | **(0.5)**  |**Our mAP** |
| :--------------------------- | :------------: | :------------------------: |:---------: |:---------: |
| Faster R-CNN++ (ResNet-101)] |      34.9      |           3.36s            |            |            |
| YOLOv2 (Darknet-19)]         |      21.6      |            25ms            |            |            |
| SSD300* (VGG16)]             |      25.1      |            22ms            |            |            |
| SSD512* (VGG16)]             |      28.8      |            53ms            |     43.1   |   42.9     |
| RetinaNet500                 |      34.4      |            90ms            |            |            |
| RFBNet300 (VGG16)            |    **29.9**    |         **15ms\***         |            |            |
| RFBNet512 (VGG16)            |    **33.8**    |         **30ms\***         |            |            |
| RFBNet512-E (VGG16)          |    **34.4**    |         **33ms\***         |            |            |


### TRAIN
python train.py --cfg_name cfgs/rfbnet/ssd_vgg16_voc_new.yml

### EVAL
python eval.py --cfg_name cfgs/rfbnet/ssd_vgg16_voc_new.yml --trained_model ../../weights/rfbnet/ssd_vgg16_voc_new/ssd_vgg16_voc_new_200000.pth
=======
# DMS_detection

>>>>>>> 80fd6302cfd500c2f6d8d94678223d6505ba7d2e
=======
# dms_classification
>>>>>>> 13e411fb210c3649955d5ed45122bc21e5be534a
