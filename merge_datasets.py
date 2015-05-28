import argparse
import scipy.io
import glob
import os
import json
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', type=str, help='Path to directory containing datasets.')
    parser.add_argument('-o', dest='output_dir', type=str, help='Path to output directory for merged dataset.')

    args = parser.parse_args()

    input_directory  = args.input_dir
    output_directory = args.output_dir
    output_json_file = '%s/merged.json' % output_directory
    output_mat_file  = '%s/merged.mat'  % output_directory

    if not os.path.exists(output_directory):
        raise RuntimeError("Output directory does not exist %s"%(output_directory))
    if not os.path.exists(input_directory):
        raise RuntimeError("%s , Directory does not exist"%(input_directory))

    json_files = glob.glob('%s/*.json' % input_directory)
    print 'Merging %d datasets: %s' % (len(json_files), json_files)

    image_id = 0
    sentence_id = 0
    images = []
    mats = []

    for json_file in json_files:
        name, _ = json_file.split('.')
        
        mat_file = '%s.mat' % name
        if not os.path.isfile(mat_file):
            raise RuntimeError("%s.mat does not exist" % name)

        # Merge JSON files
        with open(json_file, 'rb') as fh:
            dataset = json.load(fh)

            print 'Merging dataset file %s with %d images' % (json_file, len(dataset['images']))
            for image in dataset['images']:
                image['imgid'] = image_id
                sentences = image['sentences']
                image['sentids'] = []
                image['sentences'] = []
                image['dataset'] = json_file
                for sentence in sentences:
                    sentence['imgid'] = image_id
                    sentence['sentid'] = sentence_id
                
                    image['sentences'].append(sentence)
                    image['sentids'].append(sentence_id)
                    sentence_id = sentence_id + 1
                images.append(image)
                image_id = image_id + 1

        # Merge MAT files
        mat = scipy.io.loadmat(mat_file)
        mats.append(mat['feats'])

    print 'Writing JSON output to %s with %d images' % (output_json_file, len(images))
    json.dump({'images': images}, open(output_json_file, 'wb'))

    print 'Writing MAT output to: %s'  % output_mat_file
    scipy.io.savemat(output_mat_file, {'feats': np.hstack(mats)})
    