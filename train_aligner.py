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
    def __init__(self):
        super(Aligner, self).__init__()
        self.img_model = Sequential()
        self.img_model.add(Dense(4096, 1000, init='uniform'))

        self.sent_model = Sequential()
        self.sent_model.add(Bidirectional(SimpleRNN, 8856, 1000, return_sequences=True, activation='relu'))

        self.combined_model = Unsupervised()
        self.combined_model.add(Score([self.img_model, self.sent_model]))

    def _score_function(self, batch_size):
        in0     = T.matrix('regions')
        in1     = T.tensor3('words')
        location = T.imatrix('location')
        regions = T.reshape(in0, (in0.shape[0] // 20, 20, in0.shape[1]), 3)
        words   = in1.dimshuffle(0, 2, 1)
        scores  = theano.shared(np.zeros((batch_size, batch_size), dtype=np.float32))

        def _calc_scores(location, regions, words):
            x = location[0]
            y = location[1]
            score = T.sum(T.max(T.dot(regions[x], words[y]), axis=0))
            out = T.set_subtensor(scores[x, y], score)
            return (out, {scores: out})

        scores, updates = theano.scan(fn=_calc_scores, 
                                      outputs_info=None,
                                      sequences=[location],
                                      non_sequences=[regions, words])

        scores = scores[-1] # Only interested in the last step, when all scores are calculated

        return theano.function([in0, in1, location], scores, updates=updates)

    def compile(self, optimizer='sgd', loss=rank_loss, batch_size=100):
        print 'Compiling image model...'
        self.image_model = theano.function([self.img_model.get_input()], self.img_model.get_output())

        print 'Compiling sentence model...'
        self.sentence_model = theano.function([self.sent_model.get_input()], self.sent_model.get_output())

        print 'Compiling combined model...'
        self.combined_model.compile(optimizer=optimizer, loss=loss)

        print 'Compiling score function...'
        self.score_function = self._score_function(batch_size)

        print 'Done compiling!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('-b', dest='batch_size', help='Batch size', default=100, type=int)
    parser.add_argument('-e', dest='epochs', help='Number of epochs', default=1, type=int)
    parser.add_argument('-i', dest='image_in', help='Image data .h5 file', type=str)
    parser.add_argument('-s', dest='sentence_in', help='Sentence data .h5 file', type=str)
    parser.add_argument('--wo', dest='weights_out', help='Weights output file', type=str)
    parser.add_argument('--wi', dest='weights_in', default=None, help='Weights input file', type=str)

    args = parser.parse_args()

    print 'Settings:'
    print args

    # Initialize aligner
    aligner = Aligner()
    aligner.compile(batch_size=args.batch_size, optimizer=SGD(lr=0.0001, momentum=0.9), loss=dummy_loss)

    # Load data
    image_data = h5py.File(args.image_in, 'r')
    sentence_data = h5py.File(args.sentence_in, 'r')
    num_samples = 100#image_data['blobs'].shape[0]
    reading_batch_size = 4 * args.batch_size

    # Produce a list of all combinations (batch_size x batch_size)
    combinations_indexes = np.asarray(list(product(xrange(args.batch_size), xrange(args.batch_size))), dtype=np.int32)

    timings = []

    # Loop through the epochs
    for epoch in xrange(args.epochs):

        # Loop through the data
        bar = Bar('Epoch {}'.format(epoch + 1), max=len(range(num_samples)))
        for i in xrange(0, num_samples, reading_batch_size):
            img_blobs = image_data['blobs'][i : i + reading_batch_size]
            one_hots  = sentence_data['sentences/token_1hs'][i : i + reading_batch_size]
            img_blobs, one_hots = shuffle(img_blobs, one_hots, random_state=0)

            for j in xrange(0, args.batch_size, args.batch_size):
                bar.goto(i + j) # increment progress bar with increments of batch_size

                t_begin = time.clock()

                # Extract data
                d_images    = img_blobs[j : j + args.batch_size]
                d_sentences = sentence_data['sentences/token_1hs'][j : j + args.batch_size]
                d_sentences = d_sentences[:, 0, :, :]
                #d_sentences = np.ones((d_images.shape[0], 59, 8856), dtype=np.float32) # DUMMY VALUES!
                d_images    = np.reshape(d_images, (d_images.shape[0] * d_images.shape[1], d_images.shape[2]))

                # Calculate scores
                d_images_out    = aligner.image_model(d_images)
                d_sentences_out = aligner.sentence_model(d_sentences)
                scores          = aligner.score_function(d_images_out, d_sentences_out, combinations_indexes)

                # Fit image_data
                #results = aligner.combined_model.fit([[d_images], [d_sentences]], [scores])
                loss = aligner.combined_model.train_on_batch([d_images, d_sentences], scores)
                timing = time.clock() - t_begin
                print ' [Loss: {}, in {}s]'.format(loss, timing)
                if len(timings) < 10:
                    timings.append(timing)
                else:
                    print 'Average processing time: {}'.format(np.average(timings))
                    timings = []

    print 'Done training! Saving weights to {}'.format(args.weights_out)
    aligner.combined_model.save_weights(args.weights_out)
