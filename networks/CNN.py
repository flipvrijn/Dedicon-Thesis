import caffe
import skimage
import argparse
import numpy as np

from IPython import embed

class CNN(object):

    def __init__(self, args):
        caffe.set_mode_gpu()
        caffe.set_device(0)

        self.net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
        self.mean_image = '/usr/local/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

    def process(self, imgs):
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) # height * width * channel -> channel * height * width
        transformer.set_mean('data', np.load(self.mean_image).mean(1).mean(1)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        self.net.blobs['data'].reshape(20, 3, 227, 227) # assuming the best 19 regions + 1 for original image
        self.net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', x), imgs)
        out = self.net.forward()

        embed()

        activation = out['fc7']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image with CNN')
    parser.add_argument('--def', dest='prototxt', help='Network definition file')
    parser.add_argument('--net', dest='caffemodel', help='Network model file')
    parser.add_argument('-i', dest='in_image')

    args = parser.parse_args()

    cnn = CNN(args)

    image = skimage.io.imread(args.in_image)

    cnn.process([image])