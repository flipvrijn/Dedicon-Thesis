import json
import argparse
import nltk
from progress.bar import Bar
import time
import glob
import os
import cPickle
from collections import OrderedDict

def get_caption_info(image_id, captions):
    return [annotation for annotation in captions['annotations'] if annotation['image_id'] == image_id]

def get_instances_info(image_id, instances):
    return [annotation for annotation in instances['annotations'] if annotation['image_id'] == image_id]

def get_context_info(image_id, context):
    return context['images'][str(image_id)] if str(image_id) in context['images'].keys() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', dest='context_path', type=str, help='Path to the context file.')
    parser.add_argument('--instances', dest='instances_path', type=str, help='Path to the instances file.')
    parser.add_argument('--list', dest='image_list', type=str, help='Path to image list file.')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()

    print 'Loading context file %s...' % args.context_path
    context = json.load(open(args.context_path))

    print 'Loading instances file %s...' % args.instances_path
    instances = json.load(open(args.instances_path))

    with open(args.image_list, 'r') as f:
        images = f.read().split()

    im2idx = dict(zip(images, range(len(images))))

    titles = [None]*len(images)
    descriptions = [None]*len(images)
    tags = [None]*len(images)

    bar = Bar('Building context', max=len(instances['images']), suffix='%(percent)d%%')
    for image in instances['images']:
        img_filename = image['file_name']
        context_info = get_context_info(image['id'], context)

        idx = im2idx[img_filename]
        if context_info:
            titles[idx]         = context_info['title']
            descriptions[idx]   = context_info['description']
            tags[idx]           = context_info['tags']
        else:
            titles[idx]         = ''
            descriptions[idx]   = ''
            tags[idx]           = []
        bar.next()
    bar.finish()

    # save to disc
    with open(args.output_path, 'w') as f:
        cPickle.dump(titles, f)
        cPickle.dump(descriptions, f)
        cPickle.dump(tags, f)