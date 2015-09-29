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
import h5py
import time
import json
import pandas as pd
from progress.bar import Bar

from IPython import embed

# MSCOCO: 81 classes    
#CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
# Imagenet: 201 classes
CLASSES = ('__background__', 'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack', 'bagel', 'balance beam', 'banana', 'band aid', 'banjo', 'baseball', 'basketball', 'bathing cap', 'beaker', 'bear', 'bee', 'bell pepper', 'bench', 'bicycle', 'binder', 'bird', 'bookshelf', 'bow tie', 'bow', 'bowl', 'brassiere', 'burrito', 'bus', 'butterfly', 'camel', 'can opener', 'car', 'cart', 'cattle', 'cello', 'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker', 'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew', 'cream', 'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper', 'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly', 'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder', 'fig', 'filing cabinet', 'flower pot', 'flute', 'fox', 'french horn', 'frog', 'frying pan', 'giant panda', 'goldfish', 'golf ball', 'golfcart', 'guacamole', 'guitar', 'hair dryer', 'hair spray', 'hamburger', 'hammer', 'hamster', 'harmonica', 'harp', 'hat with a wide brim', 'head cabbage', 'helmet', 'hippopotamus', 'horizontal bar', 'horse', 'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala bear', 'ladle', 'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard', 'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk can', 'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck brace', 'oboe', 'orange', 'otter', 'pencil box', 'pencil sharpener', 'perfume', 'person', 'piano', 'pineapple', 'ping-pong ball', 'pitcher', 'pizza', 'plastic bag', 'plate rack', 'pomegranate', 'popsicle', 'porcupine', 'power drill', 'pretzel', 'printer', 'puck', 'punching bag', 'purse', 'rabbit', 'racket', 'ray', 'red panda', 'refrigerator', 'remote control', 'rubber eraser', 'rugby ball', 'ruler', 'salt or pepper shaker', 'saxophone', 'scorpion', 'screwdriver', 'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula', 'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer', 'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine', 'syringe', 'table', 'tape player', 'tennis ball', 'tick', 'tie', 'tiger', 'toaster', 'traffic light', 'train', 'trombone', 'trumpet', 'turtle', 'tv or monitor', 'unicycle', 'vacuum', 'violin', 'volleyball', 'waffle iron', 'washer', 'water bottle', 'watercraft', 'whale', 'wine bottle', 'zebra')

def Detect(net, image, proposals, args):
    """ Detect object classes in an image assuming the whole image is an object. """
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

    # HAX-y: When there are < args.num_regions regions, pick randomly to fill all args.num_regions spots
    if nms_boxes.shape[0] < args.num_regions:
        rand_idxs = np.random.choice(filtered_dets_boxes.shape[0], args.num_regions - nms_boxes.shape[0])
        nms_boxes = np.vstack((nms_boxes, filtered_dets_boxes[rand_idxs]))

    return nms_boxes

def vis_detections(boxes, im, rois, transformer, args):
    """ Draw detected bounding boxes. """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    classes = np.unique(boxes[:, -1:])
    print 'Visualizing {} best locations with {} distinct classes...'.format(len(boxes), len(classes))
    colormap = dict(zip(classes, plt.cm.Set1(np.linspace(0, 1, len(classes)))))

    # Visualize the best regions
    for box in boxes:
        xmin, ymin, xmax, ymax, score = box
        
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
        roi = transformer.deprocess('data', roi)
        ax.imshow(roi, interpolation='nearest', aspect='equal')
        ax.set_title('{} ({:.3f})'.format(CLASSES[int(cls)], score))

    plt.show()

def get_rois(boxes, image, transformer, args):
    """ Extracts regions of interest from image based on bounding boxes. """

    rois        = np.empty((args.num_regions + 1, 3, 224, 224), dtype=np.float32)
    rois_score  = np.empty(args.num_regions, dtype=np.float32)

    for box_idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax, score = box

        rois[box_idx]       = transformer.preprocess('data', 
            skimage.transform.resize(image[ymin:ymax, xmin:xmax], (224, 224)).astype(np.float32)
        )
        rois_score[box_idx] = score

    rois[-1] = transformer.preprocess('data',
        skimage.transform.resize(image, (224, 224)).astype(np.float32)
    )

    return rois, rois_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default=0, type=int)
    parser.add_argument('--bg', dest='ignore_background', default=True, type=bool, help='Ignore/include background')
    parser.add_argument('-i', dest='image_dir', type=str, default=None, help='Input image directory')
    parser.add_argument('--bb', dest='bounding_boxes', type=str, default=None, help='Input bounding boxes file')
    parser.add_argument('--nms_thresh', dest='nms_thresh', default=0.3, type=float, help='Non-maximum suppression threshold')
    parser.add_argument('-r', dest='num_regions', default=19, type=int, help='Number of best regions in image')
    parser.add_argument('--viz', dest='viz', default=False, type=bool, help='Visualize region localization')
    parser.add_argument('-o', dest='output', type=str, help='Output file')

    args = parser.parse_args()

    # Regional convolutional neural network model for picking the 'best' regions
    rcnn_prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'VGG_CNN_M_1024', 'test-imagenet.prototxt')
    rcnn_model    = os.path.join(cfg.ROOT_DIR, 'output', 'default', 'imagenet_train2014', 'vgg_cnn_m_1024_fast_rcnn_imagenet_iter_640000.caffemodel')

    # Convolutional neural network model for extracting 4096 feature vector from regions
    cnn_prototxt  = os.path.join(cur_path, 'models', 'VGG_ILSVRC_16_layers_deploy.prototxt')
    cnn_model     = os.path.join(cfg.ROOT_DIR, 'data', 'imagenet_models', 'VGG16.v2.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    rcnn_net = caffe.Net(rcnn_prototxt, rcnn_model, caffe.TEST)

    cnn_net  = caffe.Net(cnn_prototxt, cnn_model, caffe.TEST)
    cnn_net.blobs['data'].reshape(args.num_regions + 1, 3, 224, 224)
    transformer = caffe.io.Transformer({'data': cnn_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    # Load the image(s)
    if not os.path.isdir(args.image_dir):
        raise RuntimeError("The input '{}' needs to be a directory containing images".format(args.image_dir))
    else:
        if not args.bounding_boxes:
            raise RuntimeError("When processing a directory of images, a bounding boxes file is required as well")
        f_boxes = h5py.File(args.bounding_boxes, 'r')
        bounding_boxes, images = f_boxes['boxes'], f_boxes['images']

    print 'Processing {} images'.format(len(images))

    output = np.empty((len(images), 4), dtype=object)

    # Loading / creating features file
    start_idx = 0
    bar = Bar('Extracting boxes', max=len(images))#, suffix='%(percent)d%%')
    if os.path.isfile(args.output):
        print 'File already exists, continuing...'
        # File already exists, continuing
        f_out   = h5py.File(args.output, 'r+')
        dblobs  = f_out['blobs']
        drois   = f_out['rois']
        dscores = f_out['scores']
        dnames  = f_out['names'] 

        zeros_blob  = np.zeros((args.num_regions + 1, 4096), dtype=np.float32)
        for idx in xrange(f_out['blobs'].shape[0]):
            if np.array_equal(f_out['blobs'][idx], zeros_blob):
                start_idx = idx
                break;
            else:
                bar.next()

        print '\nContinuing with image {}...'.format(start_idx)
    else:
        # Start new file
        f_out   = h5py.File(args.output, 'w')
        dblobs  = f_out.create_dataset('blobs', (len(images), args.num_regions + 1, 4096), dtype=np.float32)
        drois   = f_out.create_dataset('rois', (len(images), args.num_regions, 4), dtype=np.float32)
        dscores = f_out.create_dataset('scores', (len(images), args.num_regions,), dtype=np.float16)
        dnames  = f_out.create_dataset('names', (len(images),), dtype=h5py.special_dtype(vlen=bytes))

        print 'Starting with the first image...'

    images = images[start_idx:]

    t_begin = time.time()
    for image_idx, image_ref in enumerate(images):
        try:
            image_idx = image_idx + start_idx # In case we are continuing

            image_name = '{}/{}'.format(args.image_dir, ''.join(chr(c) for c in f_boxes[image_ref[0]]))
            image = skimage.io.imread(image_name)

            if image.ndim != 3:
                image = np.tile(image, (3,1,1))
                image = np.transpose(image, (1,2, 0))

            generated_boxes = f_boxes[bounding_boxes[image_idx][0]].value
            generated_boxes = np.swapaxes(generated_boxes, 0, 1)

            # Detect the best locations in the image
            probable_boxes = Detect(rcnn_net, image, generated_boxes, args)

            # Extract regions of interest
            rois, rois_score = get_rois(probable_boxes, image, transformer, args)

            # Extract features from RoIs ...
            cnn_net.blobs['data'].data[...] = rois
            out = cnn_net.forward()
            dblobs[image_idx]  = out['fc7']
            drois[image_idx]   = probable_boxes[:, 0:4]
            dscores[image_idx] = rois_score
            dnames[image_idx]  = image_name

            if args.viz:
                # Visualize the detections
                vis_detections(probable_boxes, image, [rois, rois_score], transformer, args)

            if image_idx % 100 == 0 and image_idx != 0:
                print '\n{}/{} processed in {}s...'.format(image_idx, len(images), time.time() - t_begin)
                t_begin = time.time()

            bar.next()
        except ValueError:
            blob = np.empty((args.num_regions + 1, 4096), dtype=np.float32)
            blob.fill(-1)
            dblobs[image_idx] = blob
            print 'Failed to read image {} ({})'.format(image_name, image_idx)
        except:
            embed()
            raise

    bar.finish()
    f_out.close()