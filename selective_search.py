from utils.selective_search.selective_search import *

import argparse
import skimage
from IPython import embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='image', type=str, help='Input image.')

    args = parser.parse_args()

    img = skimage.io.imread(args.image)
    regions = selective_search(img)
    for v, (i0, j0, i1, j1) in regions:
        embed()