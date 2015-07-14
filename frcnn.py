import os
import sys

# Set the required paths for the F-RCNN
lib_path = os.path.join(os.path.dirname(__file__), 'networks', 'fast-rcnn', 'lib')
sys.path.insert(0, lib_path)

from fast_rcnn.config import cfg 
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, cv2
import argparse
import hdf5storage
import time
import json

from IPython import embed

#CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    embed()
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
    embed()

def Detect(net, image_path, object_proposals):
    """Detect object classes in an image assuming the whole image is an object."""
    # Load the image
    im = cv2.imread(image_path)
    h, w, c = im.shape

    # Detect all object classes and regress object bounds
    tic = time.time()
    scores, boxes = im_detect(net, im, object_proposals)
    toc = time.time()
    detect_time = (toc-tic)

    # need to process each proposal
    img_blob = {}
    img_blob['img_path'] = image_path
    img_blob['detections'] = []
    img_blob['detect_time'] = detect_time 
    # sort for each row
    sort_idxs = np.argsort(-scores, axis = 1).tolist()

    # for each proposal
    for idx, idx_rank in enumerate(sort_idxs):
        
        # get top-6
        t_boxes = []
        preds = []
        confs = [] 
        idx_rank = idx_rank[:6] # a list
        # for top-6 class
        for cls_ind in idx_rank:
            t_boxes += [ boxes[idx, 4*cls_ind:4*(cls_ind+1)].tolist() ]
            preds += [ CLASSES[cls_ind] ]
            confs += [ scores[idx, cls_ind].tolist() ] 
   
        img_blob['detections']  += [[t_boxes, preds, confs]]

    return detect_time, img_blob

def read_mat(path):
    data = hdf5storage.read(path='/', filename=path)

    return data[0][0][0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--def', dest='def', help='Network definition file')
    parser.add_argument('--net', dest='net', help='Network model file')

    args = parser.parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG16', 'test.prototxt')
    model    = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models', 'vgg16_fast_rcnn_iter_40000.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, model, caffe.TEST)

    #demo(net, 'input/bird.jpg', CLASSES)
    image_name = 'input/bird.jpg'
    box_file = os.path.join(image_name + '_boxes.mat')
    obj_proposals = read_mat(box_file)
    detect_time, img_blob = Detect(net, image_name, obj_proposals)

    embed()

    detect_json_filename = '_detections.json'
    json.dump(img_blob, open(os.path.join(detect_json_filename), 'w'))

    plt.show()
