import json
import argparse
import nltk
from progress.bar import Bar
import time
import glob
import os
import csv
from progress.bar import Bar
    
from lxml import etree

from IPython import embed

def get_instances_info(image_id, instances):
    return [annotation for annotation in instances['annotations'] if annotation['image_id'] == image_id]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synset', dest='synset_path', type=str, help='Path to the synset file.')
    parser.add_argument('--images', dest='images_path', type=str, help='Directory containing images.')
    parser.add_argument('--annotations', dest='ann_path', type=str, help='Directory containing annotations.')
    parser.add_argument('-o', dest='output_path', type=str, help='Output directory.')

    args = parser.parse_args()

    print 'Parsing synset file %s...' % args.synset_path
    with open(args.synset_path) as f_synset:
        synsets = {}
        for line in f_synset:
            splits = line.split(' ')
            synsets[splits[0].strip()] = len(synsets) + 1

    images = []
    filetypes = ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG')
    for filetype in filetypes:
        images.extend(glob.glob('%s/%s' % (args.images_path, filetype)))

    print 'Parsing annotations for %d images...' % len(images)
    dataset = []
    bar = Bar('Processing', max=len(images), suffix='%(percent)d%%')
    for image in images:
        data = {}
        filename = os.path.basename(image)

        tree = etree.parse(open('%s/%s.xml' % (args.ann_path, filename.split('.')[0])))
        data['image']   = filename
        data['width']   = int(tree.xpath('size/width/text()')[0])
        data['height']  = int(tree.xpath('size/height/text()')[0])
        data['classes'] = tree.xpath('//name/text()')
        bboxes = []
        for bbox in tree.xpath('object/bndbox'):
            box = [int(bbox.xpath('./xmin/text()')[0]), int(bbox.xpath('./ymin/text()')[0]), 
                   int(bbox.xpath('./xmax/text()')[0]), int(bbox.xpath('./ymax/text()')[0])]
            bboxes.append(box)
        data['bboxes']  = bboxes
        dataset.append(data)
        bar.next()
    bar.finish()


    print 'Writing dataset to %s...' % args.output_path
    json.dump(dataset, open('%s/imagenet_dataset_%d.json' % (args.output_path, time.time()), 'w'))

    print 'Generating data file for Caffe...'
    with open('%s/data_%d.txt' % (args.output_path, time.time()), 'wb') as train_csv:
        dwriter = csv.writer(train_csv, delimiter=' ')
        for data in dataset:
            for class_name in list(set(data['classes'])):
                dwriter.writerow([data['image'], synsets[class_name]])

    print 'Done'