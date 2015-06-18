import scipy.io
import json
from progress.bar import Bar
import numpy as np

if __name__ == '__main__':
    import h5py

    mat_path = 'fast-rcnn/data/selective_search_data/coco_test2014.mat'
    json_path = 'fast-rcnn/data/coco/annotations/instances_test2014.json'

    print 'Loading .mat file...'
    with h5py.File(mat_path) as reader:
        bar = Bar('Reading images', max=len(reader['images']))
        images = []
        for column in reader['images']:
            row_data = []
            for row_number in range(len(column)):
                row_data.append(''.join(map(unichr, reader[column[row_number]][:])))
            images.append(row_data[0])
            bar.next()
        bar.finish()

    with open(json_path) as fh:
        print 'Loading JSON file...'
        data = json.load(fh)
        
        image_id = {}
        bar = Bar('Reading IDs', max=len(data['images']))
        for image in data['images']:
            image_id[image['file_name']] = image['id']
            bar.next()
        bar.finish()

    print 'Converting images to IDs...'
    new_images = []
    bar = Bar('Converting', max=len(images))
    for image in images:
        new_images.append(image_id[image])
        bar.next()
    bar.finish()

    scipy.io.savemat('output/images.mat', {'images': new_images})