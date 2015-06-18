import json

f = 'input/flickr_data.json'

json_data = json.load(open(f, 'w'))

for image_id in json_data['images']:
    image_data = json_data['images'][image_id]
    if image_data != None:
        image_data = {}