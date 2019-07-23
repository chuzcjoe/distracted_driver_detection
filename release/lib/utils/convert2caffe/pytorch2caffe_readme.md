# ssd.pytorch_lm 分支
1 运行环境：已经ucloud-dgnet上部署环境镜像：ubuntu14_cuda8_terminal_pytorch_caffessd_xiaozhang:v0.1; 具体参数:python2.7 caffe_ssd pytorch0.3.1(如果已经编译好了caffe，直接用pip安装pytorch0.3.1)

2 保证网络最后输出要返回layer层（fc，softmax，conv等）输出。
需要增加主体网络forward函数参数：(self, x, phase='eval')，即添加一个默认参数phase，定义网络返回输出。（ssd.pytorch_lm 已经设置好了）

3 调用方法如下：
    from ConvertModel import ConvertModel_caffe
    convert2caffe = True
    save_graph = True
    #NOTE convert2caffe not need cuda
    if convert2caffe:
        save_path = './' # 保存路径
        input_shape = (1, 3, 240, 240) ＃输入图像纬度（n,c,h,w）
        model_name = str(net.__class__.__name__) + '_caffe'
        text_net, binary_weights = ConvertModel_caffe(net, input_shape, softmax=False, use_cuda=True, save_graph=save_graph)
        import google.protobuf.text_format
        with open(save_path+model_name + '.prototxt', 'w') as f:
            f.write(google.protobuf.text_format.MessageToString(text_net))
        with open(save_path+model_name + '.caffemodel', 'w') as f:
            f.write(binary_weights.SerializeToString())

4 验证
4.1 转换完后，在终端会打印pytorch网络的最后输出值。如果是ssd检测网络会输出最后一个permute之后的所有loc值（最后三行是最后一个特征层的最最终loc值，验证时对应prototxt相关的permute层）。
4.2 run_test.py 用于保存caffemodel对应层的输出。和4.1中数据对比，一致说明权重转换成功。

5 其他说明：
5.1 caffe pooling层是向上取整，因此pytorch训练网络中的pooling层需要设置ceil_mode=True, 例如：nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
5.2 由于pooling层不能获取相关具体参数，需要转换完后手动配置参数，当前转换pooling层默认参数：kernel_size＝3， stride＝2。（不一致的地方需要手动修改pooling层参数）
5.3 如果给ssd检测网络添加prior_box和detectionout 后处理层, 请查看AddPriorBoxLayer.py,只需要配置127-145行参数，会生成一个完整的prototxt。




##################
1 转换须知

1.1 caffe pooling层是向上取整，pytorch默认是向下取整。pytorch训练网络中的pooling层需要设置ceil_mode=True, 例如：nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

1.2 工具使用环境是pytorch0.3.1版本，目前使用可以最高支持到1.0保存的权重模型转换。

1.3 已经ucloud-dgnet上部署环境镜像：ubuntu14_cuda8_terminal_pytorch_caffessd_xiaozhang:v0.1; 具体参数:python2.7 caffe_ssd pytorch0.3.1

1.4 工具包：https://github.com/DG-Apollo/ssd.pytorch/tree/deepv_lm/lib/utils/convert2caffe


2 转换步骤（以resnet18为例说明转换流程）

2.1 修改网络调用函数


截图是pytorch2caffe_example下有resnet.py

（1）代码中76行，调用入口需要改为build_model()，（如果不改需要在run.py中修改调用函数名）。

（2）关注网络forwar中的最后输出，方便后续验证。



2.2 执行转换

python run.py pytorch2caffe_example/resnet.py pytorch2caffe_example/model.pth

（1）会在pytorch2caffe_example下生成对应的prototxt和caffemodel

（2）同时在终端也会输出pytorch的最后输出数据



2.3 验证caffemodel和pytorch输出是否一致

（1）由于pooling层不能获取相关具体参数，需要转换完后手动配置参数，当前转换pooling层默认参数：kernel_size＝3， stride＝2。（不一致的地方需要手动修改pooling层参数）。

（2） 执行run_test.py，并修改用于保存caffemodel对应层的输出。和2.2中数据对比，一致说明权重转换成功。





3 其他说明

3.1 ssd检测网络还需要添加prior_box和detectionout 后处理层, 请查看AddPriorBoxLayer.py,只需要配置127-145行参数，会生成一个完整的prototxt。

3.2 可以自行根据run.py调用方式嵌入到工程中：比如：https://github.com/DG-Apollo/ssd.pytorch/tree/deepv_lm


