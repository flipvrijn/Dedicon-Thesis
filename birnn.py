from theano import tensor
import theano
import json
import time
import numpy as np
import argparse

from blocks.bricks.recurrent import Bidirectional, SimpleRecurrent, GatedRecurrent
from blocks.bricks import Rectifier, Initializable, Linear
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.select import Selector
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.base import application

from collections import defaultdict
from picklable_itertools.extras import equizip
from progress.bar import Bar
from toolz import merge
from IPython import embed

class Vocab(object):

    def __init__(self, dataset_path, split_name='train'):
        self.split_name = split_name

        dataset_path = 'datasets/flickr8k.json'
        word_count_threshold = 5

        # Load dataset
        print 'Reading %s' % (dataset_path, )
        dataset = json.load(open(dataset_path, 'r'))

        # Build split
        self.split = defaultdict(list)
        for img in dataset['images']:
            self.split[img['split']].append(img)

        self.wordtoix, self.ixtoword, self.bias_init_vector = self.preProBuildWordVocab(
            self.iterSentences(self.split, self.split_name), 
            word_count_threshold)

        self.dataset = []
        for sentence in self.iterSentences(self.split, self.split_name):
            data = np.zeros((len(sentence['tokens']), len(self.ixtoword)), dtype=np.int64)
            for word_idx, w in enumerate(sentence['tokens']):
                if w in self.wordtoix:
                    data[word_idx][self.wordtoix[w]] = 1
            self.dataset.append(data)

    def sequenceLength(self):
        return len(self.dataset[0][0])

    def iterSentences(self, dataset, split = 'train'):
        for img in dataset[split]: 
            for sent in img['sentences']:
                yield sent

    def preProBuildWordVocab(self, sentence_iterator, word_count_threshold):
        # count up all word counts so that we can threshold
        # this shouldnt be too expensive of an operation
        print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
        t0 = time.time()
        word_counts = {}
        nsents = 0
        for sent in sentence_iterator:
            nsents += 1
            for w in sent['tokens']:
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print 'filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0)

        # with K distinct words:
        # - there are K+1 possible inputs (START token and all the words)
        # - there are K+1 possible outputs (END token and all the words)
        # we use ixtoword to take predicted indeces and map them to words for output visualization
        # we use wordtoix to take raw words and get their index in word vector matrix
        ixtoword = {}
        ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
        wordtoix = {}
        wordtoix['#START#'] = 0 # make first vector be the start token
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        # compute bias vector, which is related to the log probability of the distribution
        # of the labels (words) and how often they occur. We will use this vector to initialize
        # the decoder weights, so that the loss function doesnt show a huge increase in performance
        # very quickly (which is just the network learning this anyway, for the most part). This makes
        # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
        # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
        word_counts['.'] = nsents
        bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        return wordtoix, ixtoword, bias_init_vector

class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]

class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Rectifier(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence'],
                 outputs=['representation'])
    def apply(self, source_sentence):
        # Time as first dimension
        source_sentence = source_sentence.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            self.fwd_fork.apply(embeddings, as_dict=True),
            self.back_fork.apply(embeddings, as_dict=True)
        )
        return representation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',dest='dataset_path', type=str, help='Path to dataset.')
    parser.add_argument('--nhidden', dest='nhidden', default=300, type=int, help='Number of hidden units.')
    parser.add_argument('--embed', dest='embed', default=50, type=int, help='Dimension of the embedding')
    parser.add_argument('--wscale', dest='weight_scale', default=0.1, type=float, help='Initial weight scale.')

    args = parser.parse_args()

    print 'Parsing dataset file...'
    vocab = Vocab(dataset_path=args.dataset_path)

    source_sentence = tensor.lmatrix('source')
    
    encoder = BidirectionalEncoder(vocab.sequenceLength(), args.embed, args.nhidden)

    encoder.weights_init = IsotropicGaussian(args.weight_scale)
    encoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    encoder.initialize()

    print 'Parameter names: '
    enc_param_dict = Selector(encoder).get_params()
    for name, value in enc_param_dict.iteritems():
        print '    {:15}: {}'.format(value.get_value().shape, name)

    representation = encoder.apply(source_sentence)

    print 'Compiling theano function'
    f = theano.function([source_sentence], representation)

    representations = []

    bar = Bar('Encoding', max=len(vocab.dataset))
    for idx, sentence in enumerate(vocab.dataset):
        representations.append(f(sentence))
        bar.next()

        if idx == 2:
            break;
    bar.finish()

    embed()