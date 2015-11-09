import json
import flickrapi
import argparse
import time
import glob
from urlparse import urlparse
from progress.bar import Bar
import os

def fetch_flickr_data(img_url):
    """
    Fetch Flickr context data (description, title, tags) for a Flickr image URL

    Returns new data from Flickr image
    """
    # Fetch photo id from Flickr URL using the format:
    # https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}.jpg
    # 
    # More info at: https://www.flickr.com/services/api/misc.urls.html
    photo_id = os.path.split(urlparse(img_url).path)[1].split('_')[0]
    ret = flickr.photos.getInfo(photo_id=photo_id)

    if ret['stat'] == 'ok':
        for u in ret['photo']['urls']['url']:
            if u['type'] == 'photopage':
                return {
                    'url'        : u['_content'],
                    'description': ret['photo']['description']['_content'],
                    'title'      : ret['photo']['title']['_content'],
                    'tags'       : [tag['_content'] for tag in ret['photo']['tags']['tag']]
                }
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key',dest='api_key', type=str, help='Flickr API key.')
    parser.add_argument('--secret', dest='api_secret',type=str, help='Flickr API secret.')
    parser.add_argument('--instances', dest='instances_path', type=str, help='Path to the instances file.')
    parser.add_argument('--cp',dest='cp_path',help='Path to the directory where checkpoints are saved.')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()
    api_key      = args.api_key
    api_secret   = args.api_secret
    dataset_path = args.instances_path
    out_path     = args.output_path
    cp_path      = args.cp_path

    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    print 'Loading JSON instances file...'
    coco_dataset = json.load(open(dataset_path, 'r'))

    flickr_data = {}
    not_found_counter = 0

    # Load checkpoint file if it exists
    cp_file = '%s/coco_context.%d.json' % (cp_path, time.time())
    cp_files = glob.glob('%s/coco_context.*.json' % cp_path)
    if cp_files:
        cp_latest_file = sorted(cp_files)[-1]
        if os.path.isfile(cp_latest_file):
            print 'Resuming with %s...' % cp_latest_file
            flickr_data = json.load(open(cp_latest_file, 'r'))
            flickr_data = flickr_data['images']

    # For each image in the MSCOCO dataset fetch the new Flickr context
    print 'Fetching image data from Flickr...'
    bar = Bar('Fetching', max=len(coco_dataset['images']))
    for img in coco_dataset['images']:
        try:
            img_id = str(img['id'])
            # Check to skip already processed images
            if img_id not in flickr_data.keys():
                data = fetch_flickr_data(img['url'])
                flickr_data[img_id] = data

                # Image is no longer available :(
                if not data:
                    not_found_counter = not_found_counter + 1
                    print '\nERROR: Photo %s not found on Flickr! %d in total' % (img_id, not_found_counter)
            else:
                if flickr_data[img_id] is None:
                    # Image was not found in previous run
                    not_found_counter = not_found_counter + 1
                else:
                    # Check if all data is fetched in previous run
                    required_data = ['tags', 'description', 'title', 'url']
                    if len([i for i in required_data if i in flickr_data[img_id].keys()]) != len(required_data):
                        print 'Missing some data, fetching it again...'
                        data = fetch_flickr_data(img['url'])
                        flickr_data[img_id] = data

                        if not data:
                            not_found_counter = not_found_counter + 1
                            print '\nERROR: Photo %s not found on Flickr! %d in total' % (img_id, not_found_counter)
            bar.next()
        except:
            # Interrupted and saving to file to, to continue later
            print '\nInterrupted, saving to file...'
            json.dump({'images': flickr_data}, open(cp_file, 'w'))

            raise
    bar.finish()

    print 'Writing Flickr data to file...'
    json.dump({'images': flickr_data}, open(out_path, 'w'))

    print 'Cleaning up checkpoint files...'
    if cp_files:
        for f in cp_files:
            print 'Cleaning up %s' % f
            os.remove(f)