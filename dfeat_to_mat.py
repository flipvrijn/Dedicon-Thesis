import argparse
import numpy as np
from progress.bar import Bar
import scipy.io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',dest='input_file',help='Path to .dfeat file.')
    parser.add_argument('-o', dest='output_file', type=str, help='Path to output file.')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    print 'Opening dense features file %s' % input_file
    try:
        with open(input_file, 'rb') as fh:
            entries, features_size = fh.readline().split(' ')
            bar = Bar('Parsing', max=int(entries))
            feats = np.empty([int(features_size), int(entries)])
            for idx, line in enumerate(fh):
                feats[:, idx] = np.swapaxes(np.array(map(float, line.split(' '))), 0, 1)
                bar.next()
            bar.finish()
            
            print 'Writing features to %s' % output_file
            scipy.io.savemat(output_file, {'feats': feats})
    except KeyboardInterrupt:
        print 'Quitting...'