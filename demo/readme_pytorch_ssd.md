
## SSD: Single Shot MultiBox Object Detector, in PyTorch
1 环境准备：
1.1 配置anaconda, 并安装python3+（目前pytorch_ssd只支持python3）
    conda create --name your_env_name python=3.5
1.2 conda环境变量下，安装torch和torchvision，安装指定版本（0.3.1）：
    配置opencv: conda install -c https://conda.anaconda.org/menpo opencv3 (能确保安装对应版本的mkl库)
    (pip用清华源,速度快, torch=0.3.1, torchvision=0.2.1)

2 初始代码： https://github.com/amdegroot/ssd.pytorch

3 训练准备：
3.1 数据（从服务器scp）
    标注文件：机器3.105:/home/maolei/data/VOCdevkit/VOC2007/Annotations_src
    测试集：机器3.105:/home/maolei/data/VOCdevkit/VOC2007/ImageSets/Main/test_full.txt
    训练集：VOC07+VOC12的trainval.txt

3.2 pretrained 模型：https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

4 训练网络：python train.py
4.1 在train.py配置好 pretrained模型参数，模型保存路径
4.2 在train.py中注释掉：Line 84，Line 85；然后将Line 165替换为： 
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
    Hint: (每读一个epoch，next就出现‘StopIteration’异常)

4.3 在data/voc0712.py -> def pull_item(self, index)函数中，需要判断target的大小，避免训练图像没有目标对象。修改如下：
    if target.size == 0:
        img, boxes, labels = self.transform(img)
    else:
        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

5 评估网络：python eval.py
    Hint: 每次预测会生成标注文件的缓存


6 train on coco 需要的依赖包
6.1 pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    6.1.1 pre-install: cython

7 tensorboard 
环境配置：
pip install tensorboardX
pip install tensorflow (只安装cpu版本就可以)

支持可视化功能如下：
(1)priorbox可视化；
(2)augmentation可视化；
(3)训练loss曲线可视化；
(4)每一类pr曲线可视化；

tensorboard使用命令：
tensorboard --logdir ./experiments/models/ssd_voc （ip:6006 可以查看可视化结果）

训练时指定tensorboard为true，可以查看可视化(1)(2)(3):
python train.py --tensorboard true （默认log_dir='./experiments/models/ssd_voc'）

测试时可以查看可视化(4):
python eval.py 
