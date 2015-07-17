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

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
#CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def Detect(net, im, image_path, object_proposals, args):
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
    row_sort_idxs = np.argsort(-scores, axis = 1)
    # remove background regions
    if args.ignore_background:
        row_sort_idxs = row_sort_idxs[row_sort_idxs != 0].reshape(row_sort_idxs.shape[0], row_sort_idxs.shape[1] - 1)
    row_idxs_scores = scores[:,0].argsort()
    sort_idxs = row_sort_idxs[row_idxs_scores][::-1][-10:]

    sorted_scores = scores[row_idxs_scores]
    sorted_boxes  = boxes[row_idxs_scores]

    embed()

    # for each proposal
    for idx, idx_rank in enumerate(sort_idxs):
        idx_rank = idx_rank[:2] # a list

        t_box = np.empty(1, dtype=np.object)
        pred = np.empty(1, dtype=np.object)
        conf = np.empty(1, dtype=np.float)

        # TODO: select first element
        t_box = sorted_boxes[idx, 4*idx_rank[0]:4*(idx_rank[0]+1)].tolist()
        pred  = CLASSES[idx_rank[0]]
        conf  = sorted_scores[idx, idx_rank[0]]
   
        img_blob['detections']  += [[t_box, pred, conf]]

    return detect_time, img_blob

def read_mat(path):
    data = hdf5storage.read(path='/', filename=path)

    return data[0][0][0][0]

def vis_detections(im, dets, args):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for det in dets:
        box, class_name, score = det
        if score >= args.det_thresh:
            ax.add_patch(
                plt.Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(box[0], box[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')
    plt.title('{} most probable bounding boxes with {} classes'.format(len(dets), len(CLASSES)))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default=0, type=int)
    #parser.add_argument('--def', dest='def', help='Network definition file')
    #parser.add_argument('--net', dest='net', help='Network model file')
    parser.add_argument('--bg', dest='ignore_background', default=True, type=bool, help='Ignore/include background')
    parser.add_argument('--det_thresh', dest='det_thresh', default=0.001, type=float, help='Object detection threshold')

    args = parser.parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG_CNN_M_1024', 'test-coco.prototxt')
    model    = os.path.join(cfg.ROOT_DIR, 'output', 'default', 'coco_train2014', 'vgg_cnn_m_1024_fast_rcnn_iter_640000.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, model, caffe.TEST)

    # Load the image
    image_path = 'input/cars.jpg'
    im = cv2.imread(image_path)

    box_file = os.path.join(image_path + '_boxes.mat')
    obj_proposals = read_mat(box_file)
    detect_time, dets = Detect(net, im, image_path, obj_proposals, args)

    vis_detections(im, dets['detections'], args)

    plt.show()
