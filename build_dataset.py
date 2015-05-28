import argparse
import json
import nltk
import random
import os
import time
import scipy.io
from feature_extractor import *

def parse_line(line, previous_image, sentence_id):
    IID, _, sentence = line.split('\t')

    current_sentence = {
        'tokens': nltk.word_tokenize(sentence),
        'raw'   : sentence.strip(),
        'sentid': sentence_id
    }

    current_image = {}
    current_image['IID'] = IID
    current_image['sentids'] = [sentence_id]
    current_image['filename'] = '%s.jpg' % IID

    done = False

    if previous_image.keys():
        # Previous image exists
        if previous_image['IID'] == IID:
            # Still same image, different sentence
            current_image = previous_image
            current_image['sentids'].append(sentence_id)
            current_sentence['imgid'] = current_image['imgid']
            current_image['sentences'].append(current_sentence)
        else:
            # Different image
            current_image['imgid'] = previous_image['imgid'] + 1
            current_sentence['imgid'] = current_image['imgid']
            current_image['sentences'] = [current_sentence]

            done = True
    else:
        # Previous image does not exist
        current_image['imgid'] = 0
        current_sentence['imgid'] = current_image['imgid']
        current_image['sentences'] = [current_sentence]

    sentence_id = sentence_id + 1

    return (current_image, done, sentence_id)


def parse_descriptions(path):
    images = []
    current_sentence_id = 0

    with open(path, 'rb') as in_file:
        previous_image = {}
        current_image = {}
        for line in in_file:
            current_image, done, sentence_id = parse_line(line, previous_image, current_sentence_id)

            if done:
                images.append(previous_image)

            previous_image = current_image
            current_sentence_id = sentence_id
        images.append(previous_image)

    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def_path',dest='model_def_path', type=str , help='Path to the VGG_ILSVRC_16_layers model definition file.')
    parser.add_argument('--model_path', dest='model_path',type=str,  help='Path to VGG_ILSVRC_16_layers pretrained model weight file i.e VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--descriptions_path', dest='descriptions_path', type=str, help='Path to file containing descriptions of images.')
    parser.add_argument('-i',dest='input_images',help='Path to Directory containing images to be processed.')
    parser.add_argument('--WITH_GPU', action='store_true', dest='WITH_GPU', help = 'Caffe uses GPU for feature extraction')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()

    input_directory = args.input_images
    path_model_def_file = args.model_def_path
    descriptions_path = args.descriptions_path
    path_model  = args.model_path
    WITH_GPU    = args.WITH_GPU
    out_directory = args.output_path

    if not os.path.exists(out_directory):
        raise RuntimeError("Output directory does not exist %s"%(out_directory))
    
    if not os.path.exists(input_directory):
        raise RuntimeError("%s , Directory does not exist"%(input_directory))
    
    if not os.path.exists(path_model_def_file):
        raise RuntimeError("%s , Model definition file does not exist"%(path_model_def_file))
    
    if not os.path.exists(path_model):
        raise RuntimeError("%s , Path to pretrained model file does not exist"%(path_model))

    #
    # Parsing descriptions file
    #
    print 'Parsing descriptions file %s' % descriptions_path

    output = {}
    output['images'] = parse_descriptions(args.descriptions_path)

    numTrain = int(0.75  * len(output['images']))
    numTest  = int(0.125 * len(output['images']))
    numVal 	 = int(0.125 * len(output['images']))
    category_vector = [0]*numTrain + [1]*numTest + [2]*numVal
    random.shuffle(category_vector)

    print 'Splitting %d images up into %d train, %d test and %d val images' % (len(output['images']), numTrain, numTest, numVal)

    path_imgs = []

    for idx, image in enumerate(output['images']):
    	category_string = {0: 'train', 1: 'test', 2: 'val'}
    	image['split'] = category_string[category_vector[idx]]
    	output['images'][idx] = image
    	path_imgs.append(os.path.join(input_directory, image['filename'][:2], image['filename']))

    
    json_out_file = os.path.join(out_directory, 'dataset.json')
    print 'Saving descriptions JSON file to %s' % json_out_file

    json.dump(output, open(json_out_file, 'w'))
    
    #
    # Feature extraction images
    # 
    start_time = time.time()
    print "Feature Extraction for %d images starting now"%(len(path_imgs))
    feats = caffe_extract_feats(path_imgs, path_model_def_file, path_model, WITH_GPU)
    print "Total Duration for generating predictions %.2f seconds"%(time.time()-start_time)
    
    out_path = os.path.join(out_directory,'vgg_feats.mat')
    print "Saving prediction to disk %s"%(out_path)
    vgg_feats = {}
    vgg_feats['feats'] = feats
    
    scipy.io.savemat(out_path , vgg_feats)