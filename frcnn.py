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


def Detect(net, im, image_path, object_proposals):
    """Detect object classes in an image assuming the whole image is an object."""
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
        
        idx_rank = idx_rank # a list

        # get top-19
        t_boxes = np.empty(len(idx_rank), dtype=np.object)
        preds = np.empty(len(idx_rank), dtype=np.object)
        confs = np.empty(len(idx_rank), dtype=np.float)

        # for top-19 class
        for i, cls_ind in enumerate(idx_rank):
            t_boxes[i] = boxes[idx, 4*cls_ind:4*(cls_ind+1)].tolist()
            preds[i]   = CLASSES[cls_ind]
            confs[i]   = scores[idx, cls_ind]
   
        img_blob['detections']  += [[t_boxes, preds, confs]]

    return detect_time, img_blob

def read_mat(path):
    data = hdf5storage.read(path='/', filename=path)

    return data[0][0][0][0]

def vis_detections(im, dets, ignore_background=True, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for det in dets:
        boxes, class_names, scores = det
        for i in xrange(0, len(boxes)):
            if class_names[i] == '__background__' and ignore_background:
                continue
            if scores[i] >= thresh:
                ax.add_patch(
                    plt.Rectangle((boxes[i][0], boxes[i][1]),
                                  boxes[i][2] - boxes[i][0],
                                  boxes[i][3] - boxes[i][1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
                ax.text(boxes[i][0], boxes[i][1] - 2,
                        '{:s} {:.3f}'.format(class_names[i], scores[i]),
                        bbox=dict(facecolor='blue', alpha=0.5),
                            fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

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

    # Load the image
    image_path = 'input/cars.jpg'
    im = cv2.imread(image_path)

    box_file = os.path.join(image_path + '_boxes.mat')
    obj_proposals = read_mat(box_file)
    detect_time, dets = Detect(net, im, image_path, obj_proposals)

    vis_detections(im, dets['detections'], ignore_background=True, thresh=0.1)

    detect_json_filename = '_detections.json'
    json.dump(dets, open(os.path.join(detect_json_filename), 'w'))

    plt.show()
