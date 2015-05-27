import argparse
import scipy.io
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_dir', type=str, help='Path to directory containing datasets.')
    parser.add_argument('-o', dest='output_dir', type=str, help='Path to output directory for combined dataset.')

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir

    if not os.path.exists(output_directory):
        raise RuntimeError("Output directory does not exist %s"%(output_directory))
    if not os.path.exists(input_directory):
        raise RuntimeError("%s , Directory does not exist"%(input_directory))

    json_files = glob.glob('%s/*.json' % input_directory)

    images = []

    for json_file in json_files:
        name, _ = json_file.split('.')
        
        if not os.path.isfile('%s.mat' % name):
            raise RuntimeError("%s.mat does not exist" % name)

