### TRAIN
python train.py --cfg_name cfgs/rfbnet/ssd_vgg16_voc_new.yml

### EVAL
python eval.py --cfg_name cfgs/rfbnet/ssd_vgg16_voc_new.yml --trained_model ../../weights/rfbnet/ssd_vgg16_voc_new/ssd_vgg16_voc_new_200000.pth

### TEST
python test.py --cfg_name res10_face_t --job_group face --trained_model ./weights/face/res10_face_t/res10_face_t_20000dark86.9.pth --test_path ./test_imgs --vis 1

# Results
