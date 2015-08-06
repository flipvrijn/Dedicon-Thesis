import os
import sys

# Set the required paths for the F-RCNN
lib_path = os.path.join(os.path.dirname(__file__), 'networks', 'fast-rcnn', 'lib')
cur_path = os.path.dirname(__file__)
sys.path.insert(0, lib_path)

from fast_rcnn.config import cfg 
from fast_rcnn.test import im_detect
from utils.cython_nms import nms

import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import caffe
import argparse
import hdf5storage
import time
import json
import pandas as pd
import hdf5storage

from IPython import embed

# MSCOCO: 81 classes    
#CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# Imagenet: 201 classes
CLASSES = ('__background__', 'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack', 'bagel', 'balance beam', 'banana', 'band aid', 'banjo', 'baseball', 'basketball', 'bathing cap', 'beaker', 'bear', 'bee', 'bell pepper', 'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow tie', 'bow', 'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel', 'can opener', 'car', 'cart', 'cattle', 'cello', 'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker', 'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew', 'cream', 'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper', 'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly', 'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder', 'fig', 'filing cabinet', 'flower pot', 'flute', 'fox', 'french horn', 'frog', 'frying pan', 'giant panda', 'goldfish', 'golf ball', 'golfcart', 'guacamole', 'guitar', 'hair dryer', 'hair spray', 'hamburger', 'hammer', 'hamster', 'harmonica', 'harp', 'hat with a wide brim', 'head cabbage', 'helmet', 'hippopotamus', 'horizontal bar', 'horse', 'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala bear', 'ladle', 'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk can', 'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck brace', 'oboe', 'orange', 'otter', 'pencil box', 'pencil sharpener', 'perfume', 'person', 'piano', 'pineapple', 'ping-pong ball', 'pitcher', 'pizza', 'plastic bag', 'plate rack', 'pomegranate', 'popsicle', 'porcupine', 'power drill', 'pretzel', 'printer', 'puck', 'punching bag', 'purse', 'rabbit', 'racket', 'ray', 'red panda', 'refrigerator', 'remote control', 'rubber eraser', 'rugby ball', 'ruler', 'salt or pepper shaker', 'saxophone', 'scorpion', 'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine', 'syringe', 'table', 'tape player', 'tennis ball', 'tick', 'tie', 'tiger', 'toaster', 'traffic light', 'train', 'trombone', 'trumpet', 'turtle', 'tv or monitor', 'unicycle', 'vacuum', 'violin', 'volleyball', 'waffle iron', 'washer', 'water bottle', 'watercraft', 'whale', 'wine bottle', 'zebra')

def Detect(net, image, args):
    """ Detect object classes in an image assuming the whole image is an object. """
    proposals   = read_mat('{}_boxes.mat'.format(image))
    image = skimage.io.imread(image)

    print 'Detecting objects based on {} proposals...'.format(len(proposals))
    scores, boxes = im_detect(net, image, proposals)

    predictions_df = pd.DataFrame(scores, columns=CLASSES)
    del predictions_df['__background__'] # Ignore background scores
    max_scores = predictions_df.max(1) # Grab maximum scores over all regions
    max_scores.sort(ascending=False) # ... sort descending

    dets_boxes   = []
    dets_classes = []
    for i in max_scores.index:
        selected = predictions_df.iloc[i] 
        selected.sort(ascending=False) 
        cls = predictions_df.columns.get_loc(selected.index[0]) + 1 # Grab associated class
        
        dets_classes.append(cls)
        cls_boxes = boxes[i, cls*4:(cls+1)*4] # Grab associated boxes
        dets_boxes.append(np.append(cls_boxes, max_scores[i]).astype(np.float32, copy=False))

    dets_boxes   = np.array(dets_boxes)
    dets_classes = np.array(dets_classes)

    # Cap the bounding boxes to image boundary (HAX-y)
    filtered_dets_boxes = []
    for dets_box in dets_boxes:
        try:
            xmin, ymin, xmax, ymax, _ = dets_box
            roi = image[ymin:ymax, xmin:xmax]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                raise ValueError('Empty image!')
            filtered_dets_boxes.append(dets_box)
        except:
            pass
    filtered_dets_boxes = np.array(filtered_dets_boxes)

    keep         = nms(filtered_dets_boxes, args.nms_thresh) # Apply non-maximum suppression

    # Grab N 'best' locations
    nms_boxes   = filtered_dets_boxes[keep][:args.num_regions]
    nms_classes = dets_classes[keep][:args.num_regions]

    out_boxes   = np.hstack((nms_boxes, nms_classes[:, np.newaxis])).astype(np.float32, copy=False)

    return out_boxes
    
def read_mat(path):
    """ Read HDF5 .mat file format. """
    data = hdf5storage.read(path='/', filename=path)

    return data[0][0][0]

def vis_detections(boxes, image, rois, args):
    """ Draw detected bounding boxes. """
    im = skimage.io.imread(image)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    classes = np.unique(boxes[:, -1:])
    print 'Visualizing {} best locations with {} distinct classes...'.format(len(boxes), len(classes))
    colormap = dict(zip(classes, plt.cm.Set1(np.linspace(0, 1, len(classes)))))

    # Visualize the best regions
    for box in boxes:
        xmin, ymin, xmax, ymax, score, cls = box
        
        ax.add_patch(
            plt.Rectangle((xmin, ymin),
                           xmax - xmin,
                           ymax - ymin, fill=False,
                           edgecolor=colormap[cls], linewidth=3.5)
        )
        ax.text(xmin, ymin - 2,
                '{:s} {:.3f}'.format(CLASSES[int(cls)], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    # Visualize the regions of interest separately...
    fig_r, axes_r = plt.subplots(4, 6, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    fig_r.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, roi, cls, score in zip(axes_r.flat, rois[0], rois[1], rois[2]):
        ax.imshow(roi, interpolation='nearest', aspect='equal')
        ax.set_title('{} ({:.3f})'.format(CLASSES[int(cls)], score))

    plt.show()

def get_rois(boxes, image):
    """ Extracts regions of interest from image based on bounding boxes. """

    rois        = np.empty(len(boxes), dtype=object)
    rois_cls    = np.empty(len(boxes), dtype=int)
    rois_score  = np.empty(len(boxes), dtype=np.float32)

    im = skimage.io.imread(image)

    for box_idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax, score, cls = box

        rois[box_idx]       = im[ymin:ymax, xmin:xmax]
        rois_cls[box_idx]   = cls
        rois_score[box_idx] = score

    return rois, rois_cls, rois_score

def extract_cnn_features(net, img):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    # Extract features from RoIs ...
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    return out['fc7']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default=0, type=int)
    parser.add_argument('--bg', dest='ignore_background', default=True, type=bool, help='Ignore/include background')
    parser.add_argument('-i', dest='image', type=str, default=None, help='Input image')
    parser.add_argument('--nms_thresh', dest='nms_thresh', default=0.3, type=float, help='Non-maximum suppression threshold')
    parser.add_argument('-r', dest='num_regions', default=19, type=int, help='Number of best regions in image')
    parser.add_argument('--viz', dest='viz', default=False, type=bool, help='Visualize region localization')
    parser.add_argument('-o', dest='output', type=str, help='Output file')

    args = parser.parse_args()

    # Regional convolutional neural network model for picking the 'best' regions
    rcnn_prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG_CNN_M_1024', 'test-coco.prototxt')
    rcnn_model    = os.path.join(cfg.ROOT_DIR, 'output', 'default', 'coco_train2014', 'vgg_cnn_m_1024_fast_rcnn_iter_640000.caffemodel')

    # Convolutional neural network model for extracting 4096 feature vector from regions
    cnn_prototxt  = os.path.join(cur_path, 'models', 'VGG_ILSVRC_16_layers_deploy.prototxt')
    cnn_model     = os.path.join(cfg.ROOT_DIR, 'data', 'imagenet_models', 'VGG16.v2.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    rcnn_net = caffe.Net(rcnn_prototxt, rcnn_model, caffe.TEST)
    cnn_net  = caffe.Net(cnn_prototxt, cnn_model, caffe.TEST)

    # Load the image
    images = [args.image]

    print 'Processing {}'.format(images)

    output = np.empty((len(images), 4), dtype=object)

    for image_idx, image in enumerate(images):
        # Detect the best locations in the image
        print '[1/{}]: Putting {} into the R-CNN...'.format(len(images), image)
        boxes = Detect(rcnn_net, image, args)

        # Extract regions of interest
        print '[1/{}]: Extracting regions of interest...'.format(len(images))
        rois, rois_cls, rois_score = get_rois(boxes, image)

        # Grab features from locations
        print '[1/{}]: Grabbing features from regions of interest...'.format(len(images))
        
        rois_full = np.resize(rois, (rois.shape[0] + 1,))
        rois_full[-1] = skimage.io.imread(image)

        rois_cls_full = np.resize(rois_cls, (rois_cls.shape[0] + 1,))
        rois_cls_full[-1] = -1

        rois_score_full = np.resize(rois_score, (rois_score.shape[0] + 1,))
        rois_score_full[-1] = None

        v = np.empty((args.num_regions + 1, 4096))
        for roi_idx, roi in enumerate(rois_full):
            v[roi_idx] = extract_cnn_features(cnn_net, roi)
        output[image_idx][0] = v
        output[image_idx][1] = rois_full
        output[image_idx][2] = rois_cls_full
        output[image_idx][3] = rois_score_full

        if args.viz:
            # Visualize the detections
            vis_detections(boxes, image, [rois, rois_cls, rois_score], args)

    print 'Writing to {}...'.format(args.output)
    hdf5storage.savemat(args.output, {'feats': output}, format='7.3', oned_as='row', store_python_metadata=True)