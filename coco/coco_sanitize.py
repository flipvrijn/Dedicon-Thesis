import json
import argparse
import glob
import time
import os
from HTMLParser import HTMLParser
import nltk
from progress.bar import Bar

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return self.fed

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def sanitize_tags(tags):
    return [sanitize_text(tag) for tag in tags]

def sanitize_text(text):
    return ' '.join(
        [ token for token in nltk.word_tokenize(
            ''.join(
                [ fragment.replace('\n', '').strip().lower() for fragment in strip_tags(text) ]
            )
        ) if token.isalnum() or token in ['.', ',', '!', '?']]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='input_data', type=str, help='Flickr data that needs to be sanitized.')
    parser.add_argument('--cp',dest='cp_path',help='Path to the directory where checkpoints are saved.')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()
    fd  = args.input_data
    fo  = args.output_path
    cp_path = args.cp_path

    print 'Loading Flickr JSON data...'
    flickr_data = json.load(open(fd, 'r'))

    # Load checkpoint file if it exists
    cp_file = '%s/coco_sanitized.%d.json' % (cp_path, time.time())
    cp_files = glob.glob('%s/coco_sanitized.*.json' % cp_path)
    if cp_files:
        cp_latest_file = sorted(cp_files)[-1]
        if os.path.isfile(cp_latest_file):
            print 'Resuming with %s...' % cp_latest_file
            flickr_data = json.load(open(cp_latest_file, 'r'))
            flickr_data = flickr_data['images']

    bar = Bar('Sanitizing', max=len(flickr_data['images']), suffix='%(percent)d%%')
    for image_id in flickr_data['images']:
        try:
            image_data = flickr_data['images'][image_id]
            if image_data != None and 'sanitized' not in image_data.keys():
                # Check if all data is available
                image_data = {
                    'tags'          : sanitize_tags(image_data['tags']),
                    'description'   : sanitize_text(image_data['description']),
                    'title'         : sanitize_text(image_data['title']),
                    'url'           : image_data['url'],
                    'sanitized'     : True
                }

                flickr_data[image_id] = image_data
            bar.next()
        except:
            # Interrupted and saving to file to, to continue later
            print '\nInterrupted, saving to file...'
            json.dump({'images': flickr_data}, open(cp_file, 'w'))

            raise
    bar.finish()

    print 'Writing Flickr data to file...'
    json.dump({'images': flickr_data}, open(fo, 'w'))

    print 'Cleaning up checkpoint files...'
    if cp_files:
        for f in cp_files:
            print 'Cleaning up %s' % f
            os.remove(f)