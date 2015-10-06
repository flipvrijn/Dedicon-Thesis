import caffe
import argparse
import numpy as np
import time
from progress.bar import Bar

from scipy.sparse import csr_matrix, vstack

import skimage
import skimage.transform
import skimage.io

from PIL import Image

from IPython import embed

def load_image(filename, cnn_mean, resize=256, crop=224):
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
    data = data[(2, 1, 0), :, :]
    mean = np.asarray([103.939, 116.779, 123.68])
    mean = mean[:, np.newaxis, np.newaxis]
    data -= mean

    return data

def main(args):
    image_path = args.image_dir.rstrip('/')

    # Use GPU processing, because faster = better
    caffe.set_mode_gpu()

    # Construct CNN from input files
    cnn = caffe.Net(args.prototxt, args.model, caffe.TEST)

    # Setup CNN such that it can process n_samples = batch_size
    cnn.blobs['data'].reshape(args.batch_size, 3, 224, 224)

    failed_images = []
    for split_name in ['train', 'val']:
        with open('{}/{}2014list.txt'.format(image_path, split_name)) as f:
            file_data = f.read()
            lines = file_data.split()

            average_per_batch = []

            bar = Bar('Processing {}'.format(split_name), max=len(lines))
            for i in xrange(0, len(lines), args.batch_size):
                bar.goto(i)

                time_b = time.clock() # Start timing

                image_files = lines[i : i + args.batch_size] # Image filenames for this batch

                # Preprocessing the batch of images
                cnn_in = np.zeros((args.batch_size, 3, 224, 224), dtype=np.float32)
                for img_idx, img in enumerate(image_files):
                    try:
                        cnn_in[img_idx, :] = load_image('{}/{}/{}'.format(image_path, split_name, img), args.mean)
                    except:
                        # Image is corrupt or missing
                        failed_images.append('{}/{}'.format(split_name, img))
                        print 'Image {} failed.'.format(img)
                out = cnn.forward_all(blobs=['conv5_4'], **{'data': cnn_in})

                # Get features from CNN
                feat = cnn.blobs['conv5_4'].data
                if len(image_files) < args.batch_size:
                    feat = feat[len(image_files), :]

                # Store it in a sparse matrix
                if i == 0:
                    feat_flatten_list = csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
                else:
                    feat_flatten_list = vstack([feat_flatten_list, csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

                average_per_batch.append(time.clock() - time_b) # Book-keeping

                if i % 1000 == 0 and i != 0:
                    print '\n{}s per batch'.format(np.mean(average_per_batch))
                    average_per_batch = [] # Empty it again

            bar.finish()

        # Save it to an .npz file for later use
        out = {
            'data': feat_flatten_list.data,
            'indices': feat_flatten_list.indices,
            'indptr': feat_flatten_list.indptr,
            'shape': feat_flatten_list.shape
        }

        print 'Saving to {}/{}.npz'.format(args.out_dir, split_name)
        np.savez('{}/{}.npz'.format(args.out_dir, split_name), **out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for the MSCOCO dataset')

    parser.add_argument('--proto', dest='prototxt', help='Deploy prototxt file for CNN', type=str)
    parser.add_argument('--model', dest='model', help='Caffemodel file for CNN', type=str)
    parser.add_argument('--imgs', dest='image_dir', help='Input image directory', type=str)
    parser.add_argument('--mean', dest='mean', help='Mean file', type=str)
    parser.add_argument('-o', dest='out_dir', help='Output directory', type=str)
    parser.add_argument('-b', dest='batch_size', default=50, type=int, help='CNN batch size')

    args = parser.parse_args()

    main(args)