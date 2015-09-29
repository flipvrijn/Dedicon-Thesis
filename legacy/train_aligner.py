import sys
import theano
import theano.tensor as T
import numpy as np
import argparse
import h5py
import time
from itertools import product
from progress.bar import Bar
from sklearn.utils import shuffle

sys.path.insert(0, '/home/flipvanrijn/Workspace/Dedicon-Thesis/')

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import Bidirectional, SimpleRNN
from keras.layers import containers
from keras.optimizers import SGD

from networks.keras.score import Score # Custom merge layer
from networks.keras.unsupervised import Unsupervised
from networks.keras.customloss import *

from IPython import embed

class Aligner(object):
    """Sentence image aligner"""
    def __init__(self, batch_size):
        super(Aligner, self).__init__()
        self.batch_size = batch_size

        self.img_model = Sequential()
        self.img_model.add(Dense(4096, 1000, init='uniform'))

        self.sent_model = Sequential()
        self.sent_model.add(Bidirectional(SimpleRNN, 8856, 1000, return_sequences=True, activation='relu'))

        self.combined_model = Unsupervised()
        self.combined_model.add(Score([self.img_model, self.sent_model], self.batch_size))

    def compile(self, optimizer='sgd', loss=rank_loss, theano_mode=None):
        print 'Compiling combined model...'
        self.combined_model.compile(optimizer=optimizer, loss=loss, theano_mode=theano_mode)

        print 'Done compiling!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('-b', dest='batch_size', help='Batch size', default=100, type=int)
    parser.add_argument('-e', dest='epochs', help='Number of epochs', default=1, type=int)
    parser.add_argument('-i', dest='image_in', help='Image data .h5 file', type=str)
    parser.add_argument('-s', dest='sentence_in', help='Sentence data .h5 file', type=str)
    parser.add_argument('--wo', dest='weights_out', help='Weights output file', type=str)
    parser.add_argument('--wi', dest='weights_in', default=None, help='Weights input file', type=str)
    parser.add_argument('--debug', dest='debug', default=False, help='Enable debug before training', type=bool)
    parser.add_argument('--cp', dest='cp_threshold', default=10000, help='Checkpoint after this many samples', type=int)
    parser.add_argument('--co', dest='cp_out', help='Output directory for checkpoints', type=str)

    args = parser.parse_args()

    print 'Settings:'
    print args

    # Initialize aligner
    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if (not isinstance(output[0], np.random.RandomState) and
                np.isnan(output[0]).any()):
                #orig_stdout = sys.stdout 
                #f = file('out.txt', 'w')
                #sys.stdout = f

                print '*** NaN detected ***'
                embed()
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [np.asarray(input[0]) for input in fn.inputs]
                print 'Outputs: %s' % [np.asarray(output[0]) for output in fn.outputs]

                #sys.stdout = orig_stdout
                #f.close()
                break
    aligner = Aligner(batch_size=args.batch_size)
    aligner.compile(optimizer=SGD(lr=0.0001, momentum=0.9), theano_mode=theano.compile.MonitorMode(
        post_func=detect_nan
        ))

    # Load data
    image_data = h5py.File(args.image_in, 'r')
    sentence_data = h5py.File(args.sentence_in, 'r')
    num_samples = image_data['blobs'].shape[0]
    reading_batch_size = 20 * args.batch_size

    timings = []

    if args.debug:
        embed()

    # Loop through the epochs
    for epoch in xrange(args.epochs):

        # Loop through the data
        bar = Bar('Epoch {}'.format(epoch + 1), max=len(range(num_samples)))
        for i in xrange(0, num_samples, reading_batch_size):
            img_blobs = image_data['blobs'][i : i + reading_batch_size]
            one_hots  = sentence_data['sentences/token_1hs'][i : i + reading_batch_size]
            img_blobs, one_hots = shuffle(img_blobs, one_hots, random_state=0)
            
            for j in xrange(0, reading_batch_size, args.batch_size):
                bar.goto(i + j) # increment progress bar with increments of batch_size

                t_begin = time.clock()

                # Extract data
                d_images    = img_blobs[j : j + args.batch_size]
                d_sentences = sentence_data['sentences/token_1hs'][j : j + args.batch_size]
                d_sentences = d_sentences[:, 0, :, :]
                d_images    = np.reshape(d_images, (d_images.shape[0] * d_images.shape[1], d_images.shape[2]))

                # Fit image_data
                #results = aligner.combined_model.fit([[d_images], [d_sentences]], [scores])
                loss = aligner.combined_model.train_on_batch([d_images, d_sentences])
                timing = time.clock() - t_begin
                print ' [Loss: {}, in {}s]'.format(loss, timing)
                if len(timings) < 10:
                    timings.append(timing)
                else:
                    print 'Average processing time: {}'.format(np.average(timings))
                    timings = []

                if (i + j) % args.cp_threshold == 0:
                    aligner.combined_model.save_weights('{}/aligner_cp{}.h5'.format(args.cp_out, time.time()))

    print 'Done training! Saving weights to {}'.format(args.weights_out)
    aligner.combined_model.save_weights(args.weights_out)
