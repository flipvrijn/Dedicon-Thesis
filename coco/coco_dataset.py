import json
import argparse
import nltk
from progress.bar import Bar
import time
import glob
import os

def get_caption_info(image_id, captions):
    return [annotation for annotation in captions['annotations'] if annotation['image_id'] == image_id]

def get_instances_info(image_id, instances):
    return [annotation for annotation in instances['annotations'] if annotation['image_id'] == image_id]

def get_context_info(image_id, context):
    return context['images'][str(image_id)] if str(image_id) in context['images'].keys() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', dest='context_path', type=str, help='Path to the context file.')
    parser.add_argument('--captions', dest='captions_path', type=str, help='Path to the captions file.')
    parser.add_argument('--instances', dest='instances_path', type=str, help='Path to the instances file.')
    parser.add_argument('--cp',dest='cp_path',help='Path to the directory where checkpoints are saved.')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()

    print 'Loading context file %s...' % args.context_path
    context = json.load(open(args.context_path))

    print 'Loading captions file %s...' % args.captions_path
    captions = json.load(open(args.captions_path))

    print 'Loading instances file %s...' % args.instances_path
    instances = json.load(open(args.instances_path))

    """
    image_struct = {
        'filename'  : None,
        'date_captured': None,
        'height'    : None,
        'width'     : None,
        'image_id'  : None,
        'sentences' : [],
        'context'   : None,
        'sent_ids'  : [],
        'split'     : None,
        'instances' : []
    }

    context_struct = {
        'image_id'  : None,
        'context_id': None,
        'tags'      : [],
        'description': None,
        'title'     : None,
        'url'       : None
    }

    sentence_struct = {
        'image_id'  : None,
        'sent_id'   : None,
        'raw'       : None,
        'tokens'    : []
    }
    """

    output = {
        'images': []
    }

    # Load checkpoint file if it exists
    cp_file = '%s/coco_data.%d.json' % (args.cp_path, time.time())
    cp_files = glob.glob('%s/coco_data.*.json' % args.cp_path)
    if cp_files:
        cp_latest_file = sorted(cp_files)[-1]
        if os.path.isfile(cp_latest_file):
            print 'Resuming with %s...' % cp_latest_file
            output = json.load(open(cp_latest_file, 'r'))

    context_id = 0

    bar = Bar('Merging', max=len(instances['images'])) # , suffix='%(percent)d%%'
    for image in instances['images']:
        try:
            caption_info_list = get_caption_info(image['id'], captions)

            # Construct sentence structs and sentence ids list from each caption
            sentences = []
            sent_ids = []
            for caption_info in caption_info_list:
                sentence_struct = {
                    'image_id'  : image['id'],
                    'sent_id'   : caption_info['id'],
                    'raw'       : caption_info['caption'],
                    'tokens'    : nltk.word_tokenize(caption_info['caption'])
                }
                sentences.append(sentence_struct)
                sent_ids.append(caption_info['id'])

            # Construct context struct
            context_info = get_context_info(image['id'], context)

            context_struct = None
            if context_info:
                context_struct = {
                    'image_id'  : image['id'],
                    'context_id': context_id,
                    'tags'      : context_info['tags'],
                    'description': context_info['description'],
                    'title'     : context_info['title'],
                    'url'       : context_info['url']
                }

            # Finally, construct the image struct using all information
            image_struct = {
                'filename'  : image['file_name'],
                'date_captured': image['date_captured'],
                'height'    : image['height'],
                'width'     : image['width'],
                'image_id'  : image['id'],
                'sentences' : sentences,
                'context'   : context_struct,
                'sent_ids'  : sent_ids,
                'split'     : None,
                'instances' : get_instances_info(image['id'], instances)
            }

            context_id += 1

            output['images'].append(image_struct)

            bar.next()
        except:
            # Interrupted and saving to file to, to continue later
            print '\nInterrupted, saving to file...'
            json.dump(output, open(cp_file, 'w'))
    bar.finish()