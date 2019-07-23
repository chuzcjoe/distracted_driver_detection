#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
#[[ $# -eq 0 ]] && echo "$0 model resize_scale" && exit 0
#model=$1
#resize_scale=$2

#echo ${model}, "zz", ${resize_scale}

model="test"
resize_scale=0.5

d=`pwd`
current_dir=`basename $d`    #get current dir
iter=`echo $model |awk -F '.' '{print $1}' |awk -F '_' '{print $NF}'`
save_dir=ssd_${current_dir}_iter_$iter
#mkdir -p $save_dir    #create dir

anno_key='poi_water'
data_path=/train/trainset/1/DMS/
pred_file=/train/trainset/1/DMS/results/det_test_poi_water.txt
test_file=/train/trainset/1/DMS/test/new_test.txt
save_dir=/train/trainset/1/DMS/

#anno_key='poi_phone'
#data_path=/train/trainset/1/DMS/
#pred_file=/train/trainset/1/DMS/results/det_test_poi_phone.txt
#test_file=/train/trainset/1/DMS/test/new_test.txt
#save_dir=/train/trainset/1/DMS/



#anno_key='poi_palm'
#data_path=/train/trainset/1/DMS/
#pred_file=/train/trainset/1/DMS/results/det_test_poi_palm.txt
#test_file=/train/trainset/1/DMS/test/new_test.txt
#save_dir=/train/trainset/1/DMS/


#anno_key='poi_face'
#data_path=/train/trainset/1/DMS/
#pred_file=/train/trainset/1/DMS/results/det_test_poi_face.txt
#test_file=/train/trainset/1/DMS/test/new_test.txt
#save_dir=/train/trainset/1/DMS/


echo "begin roc task"

#--scale_ranges="(40,720)"
python $SHELL_FOLDER/roc.py --pred_file=$pred_file --test_file=$test_file --data_path=$data_path --save_dir=$save_dir --anno_key=$anno_key --ious=0.5
