from feature_extractor import * 
import argparse
import os
import time
import scipy.io

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def_path',dest='model_def_path', type=str , help='Path to the VGG_ILSVRC_16_layers model definition file.')
    parser.add_argument('--model_path', dest='model_path',type=str,  help='Path to VGG_ILSVRC_16_layers pretrained model weight file i.e VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('--filter',default = None ,dest='filter', help='Text file containing images names in the input directory to be processed. If no argument provided all images are processed.')
    parser.add_argument('-i',dest='input_directory',help='Path to Directory containing images to be processed.')
    parser.add_argument('--WITH_GPU', action='store_true', dest='WITH_GPU', help = 'Caffe uses GPU for feature extraction')
    parser.add_argument('-o', dest='output_path', type=str, help='Path to output file for dataset.')

    args = parser.parse_args()

    input_directory = args.input_directory
    filter_path = args.filter
    path_model_def_file = args.model_def_path
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

    if not filter_path == None:
        imgs = open(filter_path,'r').read().splitlines()
    else:
        imgs = os.listdir(input_directory)

    path_imgs = [ os.path.join(input_directory , file) for file in imgs ]

    start_time = time.time()
    print "Feature Extraction for %d images starting now"%(len(path_imgs))
    feats = caffe_extract_feats(path_imgs, path_model_def_file, path_model, WITH_GPU)
    print "Total Duration for generating predictions %.2f seconds"%(time.time()-start_time)
    
    out_path = os.path.join(out_directory,'vgg_feats.mat')
    print "Saving prediction to disk %s"%(out_path)
    vgg_feats = {}
    vgg_feats['feats'] = feats
    
    scipy.io.savemat(out_path , vgg_feats)