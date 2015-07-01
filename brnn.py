import json
import time
from collections import defaultdict, OrderedDict
import numpy as np

from IPython import embed
from pybrain.structure.networks import BidirectionalNetwork
from pybrain.structure.modules import ReluLayer

class Vocab(object):

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

        self.dataset = []
        for sentence in self.iterSentences(self.split, self.split_name):
            data = []
            for w in sentence['tokens']:
                if w in self.wordtoix:
                    word_vec = np.zeros(len(self.ixtoword))
                    word_vec[self.wordtoix[w]] = np.intc(1)
                    data.append(word_vec)
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

if __name__ == '__main__':
    vocab = Vocab()

    print 'Building network...'
    n = BidirectionalNetwork(seqlen=vocab.sequenceLength(), inputsize=1, hiddensize=600)
    n.componentclass = ReluLayer
    n.outcomponentclass = ReluLayer

    embed()