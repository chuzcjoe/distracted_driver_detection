import sys
import caffe, cv2, numpy as np

def save_output_txt(net, name, path):
    print(name, net.blobs[name].data.shape) #np.array
    ary = net.blobs[name].data
    flat = ary.reshape(-1)
    datasize = ary.size
    name = name.replace('/', '_')
    f = open(path+'/{}.txt'.format(name), 'w')
    for i in range(int((datasize + 3) / 4)):
        maxj = min((datasize - i * 4), 4)
        for j in range(maxj):
            f.write(str(flat[i * 4 + j]) + " ")
        f.write("\n")
    f.close()

#TODO user also can add 'w h' params
if len(sys.argv) < 3:
    print("Hint : python run.py torch_net_name torch_model_path\n")

deploy_file = sys.argv[1]
trained_model = sys.argv[2]

caffe.set_mode_cpu()
net = caffe.Net(deploy_file, trained_model, caffe.TEST)


# image_resize = [224, 224] #H, W
# img = cv2.imread('/home/maolei/data/xiaozhang/g_1.jpg')
# img = cv2.resize(img, (224, 224))

input_shape = (224, 224, 3) #H, W, C
img = np.zeros(input_shape).astype('uint8')
img_ary = np.asarray(img[:,:]).astype(float)
img_ary = np.transpose(img_ary, (2, 0, 1))
# mean = np.array([128., 128., 128.])#np.array([104., 117., 123.])#TODO
# mean = np.reshape(mean, (3, 1, 1))
# mean = np.broadcast_to(mean, (3, image_resize[1], image_resize[0]))
# img_ary = np.subtract(img_ary, mean)
# img_ary *= 0.01
net.blobs['data'].data[...] = img_ary
# ann = np.loadtxt('/home/devymex/data/ssd/dump/zhicheng/annotations.txt')
# net.blobs['label'].data[...] = ann

# params = net.params.keys()
# print(params)

out = net.forward()
path = './'
save_output_txt(net, 'Softmax_1', path)
#save_output_txt(net, 'ConvNdBackward21', path)


# layer {
#   name: "ConvNd_18_flat"
#   type: "Flatten"
#   bottom: "ConvNd_18"
#   top: "ConvNd_18_flat"
#   flatten_param {
#     axis: 1
#   }
# }

