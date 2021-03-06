from utils.selective_search.selective_search import *

import argparse
import skimage
import glob
import sys
import hdf5storage
import os
import numpy as np
import scipy.io

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

from progress.bar import Bar

def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors

def visualize(img, regions, args):
    if args.style == 0:
        # Bounding boxes visualization
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        plt.imshow(img)
        for v, (y0, x0, y1, x1) in regions:
            ax.add_patch(
                patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', fill=False)
            )
        plt.show()
    else:
        # Hierarchical segmentation visualization
        fig, ax = plt.subplots()
        slider_ax = fig.add_axes([0.2, 0.03, 0.65, 0.03])

        (R, F, L) = hierarchical_segmentation(img)

        colors = generate_color_table(R)
        results = []
        for depth, label in enumerate(F):
            result = colors[label]
            result = result.astype(numpy.uint8)
            results.append(result)

        sresults = Slider(slider_ax, 'Depth', 0, len(results) - 1, valinit=0, valfmt='%0.0f')

        def update(val):
            result_id = sresults.val
            ax.imshow(results[int(result_id)])
        sresults.on_changed(update)

        ax.imshow(results[0])

        plt.show()

if __name__ == '__main__':
    """
    Generates and visualizes regions 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='single_image_path', type=str, default=None, help='Path to single image file.')
    parser.add_argument('-i',dest='image_path', type=str, default=None, help='Directory containing images.')
    parser.add_argument('-o',dest='output_path', type=str, default=None, help='Path to output file.')
    parser.add_argument('--viz', dest='viz', type=int, default=0, help='Visualize selective search')
    parser.add_argument('--style', dest='style', type=int, default=0, help='Visualization style [0=bounding boxes, 1=segmentation]')

    args = parser.parse_args()
    print args
    viz = args.viz
    images = []
    if not args.single_image_path:
        filetypes = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG')
        for filetype in filetypes:
            images.extend(glob.glob('%s/%s' % (args.image_path, filetype)))
    else:
        images = [args.single_image_path]

    if args.single_image_path:
        output_path = args.output_path if args.output_path else args.single_image_path + '_boxes.mat'
    elif not args.single_image_path and not args.output_path:
        raise RuntimeError("No output file specified!")

    # If images are available in input path
    if len(images):
        image_names = [None for i in xrange(len(images))] # Holds the image filenames
        boxes       = np.empty(len(images), dtype=np.object) # Holds the bounding boxes per image

        bar = Bar('Searching boxes', max=len(images))#, suffix='%(percent)d%%')
        for image_idx, image in enumerate(images):
            img = skimage.io.imread(image)
            try:
                regions = selective_search(img, color_spaces=['rgb', 'nrgb', 'hue'], ks=[100, 200], feature_masks=[(0,1,1,0)])

                # Show the visualization and ask whether to continue with visualizing
                if viz:
                    visualize(img, regions, args)
                    sys.stdout.write('Keep visualizing? [y/n]: ')
                    choice = raw_input().lower()
                    if choice != 'y':
                        viz = False

                # Put all regions of the image in a collection
                boxes_image = np.zeros([len(regions), 4], dtype=np.double)
                for idx_region, data in enumerate(regions):
                    _, (y0, x0, y1, x1) = data
                    # compat: [y0, x0, y1, x1] = [y, x, height, width]
                    boxes_image[idx_region] = [x0, y0, x1, y1]

                # Store the bounding boxes in the collection
                boxes[image_idx] = boxes_image
                # Store the image filename in the collection
                image_names[image_idx] = os.path.basename(image)

                bar.next()
            except:
                print '{} failed to process...'.format(image)
                pass
        bar.finish()
        
        print 'Writing to file %s...' % output_path
        hdf5storage.savemat(output_path, {'images': image_names, 'boxes': boxes}, format='7.3', oned_as='row', store_python_metadata=True)
    else:
        print 'Could not find any images in %s' % args.image_path

