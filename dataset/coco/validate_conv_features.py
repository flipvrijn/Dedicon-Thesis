import caffe
import argparse
import numpy as np
import time
import os.path
from progress.bar import Bar

from scipy.sparse import csr_matrix, vstack

import skimage
import skimage.transform
import skimage.io

from PIL import Image

import matplotlib.pyplot as plt

from IPython import embed

# From author Kelvin Xu:
# https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
def load_image(filename, resize=256, crop=224):
    image = Image.open(filename)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width  - crop) / 2
    top  = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)

    data = data.transpose((2, 0, 1))
    
    mean = np.asarray([103.939, 116.779, 123.68])
    mean = mean[:, np.newaxis, np.newaxis]
    data -= mean
    data = data[(2, 1, 0), :, :]
    return data

def main(args):
    image_path = args.image_dir.rstrip('/')

    # Use GPU processing, because faster = better
    caffe.set_mode_gpu()

    # Construct CNN from input files
    cnn = caffe.Net(args.prototxt, args.model, caffe.TEST)

    # Setup CNN such that it can process n_samples
    cnn.blobs['data'].reshape(1, 3, 224, 224)

    # Loading pre-computed dataset
    print 'Loading pre-computed dataset'
    loader = np.load(args.dataset)
    dataset_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

    print 'Loading images list'
    with open(os.path.join(args.image_dir, 'train2014list.txt'), 'r') as f:
        filenames = f.read().split()

    assert len(filenames) == dataset_matrix.shape[0]

    sample_indices = np.random.choice(range(dataset_matrix.shape[0]), args.samples)
    success = 0
    failed  = 0

    bar = Bar('Sampling...', max=args.samples)
    for i in sample_indices:
        img = filenames[i]

        out = cnn.forward_all(blobs=['conv5_4'], **{'data': load_image(os.path.join(image_path, 'train', img))})
        feat = cnn.blobs['conv5_4'].data
        feat = feat.transpose((0, 2, 3, 1))
        feat_flatten = feat.flatten()

        # Compare them and tally
        if np.all(np.equal(feat_flatten, dataset_matrix[i])):
            success += 1
        else:
            failed += 1

        bar.next()
    bar.finish()

    print 'Results: {} success, {} failed, {} total'.format(success, failed, len(sample_indices))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate feature extraction based on pre-computed dataset')

    parser.add_argument('--proto', dest='prototxt', help='Deploy prototxt file for CNN', type=str)
    parser.add_argument('--model', dest='model', help='Caffemodel file for CNN', type=str)
    parser.add_argument('--imgs', dest='image_dir', help='Input image directory', type=str)
    parser.add_argument('-d', dest='dataset', help='Pre-computed dataset', type=str)
    parser.add_argument('-b', dest='samples', default=50, type=int, help='Number of samples to validate')

    args = parser.parse_args()

    main(args)