import json
import time
import numpy as np
from collections import defaultdict, OrderedDict
from IPython import embed
from progress.bar import Bar
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.algorithms import GradientDescent, Adam
from blocks.main_loop import MainLoop
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Rectifier, Linear, Softmax
from blocks.bricks.lookup import LookupTable

from fuel.datasets import Dataset
from fuel.transformers import Mapping, Batch, Padding
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

class Vocab(Dataset):

    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, split_name='train', **kwargs):
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

        super(Vocab, self).__init__()

    def open(self):
        return self.iterSentences(self.split, self.split_name)

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError

        current_state = next(state)

        data = []
        for w in current_state['tokens']:
            if w in self.wordtoix:
                word_vec = np.zeros(len(self.ixtoword))
                word_vec[self.wordtoix[w]] = np.intc(1)
                data.append(word_vec)
        embed()
        return (data,)
        

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


class Network(object):

    def __init__(self, vocab, dim):
        self.vocab = vocab
        self.dim = dim

        self.network = Bidirectional(SimpleRecurrent(dim, Rectifier()),
            weights_init=IsotropicGaussian(0.01),
            biases_init=Constant(vocab.bias_init_vector))

        self.data_stream = self.vocab.get_example_stream()
        self.data_stream = Mapping(self.data_stream, self.make_target, add_sources=('target',))
        self.data_stream = Batch(self.data_stream, iteration_scheme=ConstantScheme(100))

    def initialize(self):
        self.network.initialize()

    def make_target(self, data):
        return data

    def components(self):
        features = tensor.lmatrix('features')
        target = tensor.lmatrix('target')
        lookup = LookupTable(len(vocab.wordtoix), dim,
            weights_init=IsotropicGaussian(0.01),
            biases_init=Constant(0.))
        lookup.initialize()
        y_hat = network.network.apply(lookup.apply(features))

        linear = Linear(2 * self.dim, len(vocab.wordtoix),
            weights_init=IsotropicGaussian(0.01),
            biases_init=Constant(0.))
        linear.initialize()
        y_hat = linear.apply(y_hat)

        seq_length = y_hat.shape[0]
        batch_size = y_hat.shape[1]
        y_hat = Softmax().apply(y_hat.reshape((seq_length * batch_size, 1))).reshape(y_hat.shape)

        cost = CategoricalCrossEntropy().apply(
            target.flatten(),
            y_hat.reshape((-1, len(vocab.wordtoix)))) * seq_length * batch_size
        cost.name = 'cost'
        cg = ComputationGraph([cost])
        model = Model(cost)
        algorithm = GradientDescent(step_rule=Adam(), cost=cost, params=cg.parameters)

        return (model, algorithm, self.data_stream)

if __name__ == '__main__':
    dim = 100
    vocab = Vocab()

    network = Network(vocab, dim)
    network.initialize()
    model, algorithm, data_stream = network.components()

    embed()

    main_loop = MainLoop(model=model, algorithm=algorithm, data_stream=data_stream)

    main_loop.run()

    #rnn = network(vocab_bias_vector=vocab[2])
    #rnn.initialize()