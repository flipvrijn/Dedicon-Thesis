import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

if __name__ == '__main__':
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'imagenet_models', 'VGG16.v2.caffemodel')

    caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    demo(net, '000004', ('car',))