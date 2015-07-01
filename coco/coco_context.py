import json
import flickrapi
from urlparse import urlparse
from progress.bar import Bar
import os

def fetch_flickr_data(img_url):
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

api_key      = u'0a5e2d1908aaf4580a0b1c4e430a6838'
api_secret   = u'354c7458a637c929'
dataset_path = 'fast-rcnn/data/coco/annotations/instances_train2014.json'
out_path     = 'output/flickr_data.json'
cp_path = 'output/flickr_data.cp.json'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

print 'Loading JSON annotations file...'
coco_dataset = json.load(open(dataset_path, 'r'))

flickr_data = {}
not_found_counter = 0

if os.path.isfile(cp_path):
    print 'Loading checkpoint file...'
    flickr_data = json.load(open(cp_path, 'r'))
    flickr_data = flickr_data['images']

print 'Fetching image data from Flickr...'
bar = Bar('Fetching', max=len(coco_dataset['images']))
for img in coco_dataset['images']:
    try:
        img_id = str(img['id'])
        if img_id not in flickr_data.keys():
            data = fetch_flickr_data(img['url'])
            flickr_data[img_id] = data

            if not data:
                not_found_counter = not_found_counter + 1
                print '\nERROR: Photo %s (%s) not found on Flickr! %d in total' % (photo_id, img_id, not_found_counter)
        else:
            if flickr_data[img_id] is None:
                not_found_counter = not_found_counter + 1
            else:
                required_data = ['tags', 'description', 'title', 'url']
                if len([i for i in required_data if i in flickr_data[img_id].keys()]) != len(required_data):
                    print 'Missing some data, fetching it again...'
                    data = fetch_flickr_data(img['url'])
                    flickr_data[img_id] = data

                    if not data:
                        not_found_counter = not_found_counter + 1
                        print '\nERROR: Photo %s (%s) not found on Flickr! %d in total' % (photo_id, img_id, not_found_counter)
        bar.next()
    except:
        # Interrupted and saving to file to, to continue later
        print '\nInterrupted, saving to file...'
        json.dump({'images': flickr_data}, open(cp_path, 'w'))

        raise
bar.finish()

print 'Writing Flickr data to file...'
json.dump({'images': flickr_data}, open(out_path, 'w'))