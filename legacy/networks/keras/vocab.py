import numpy
from utils import *
from itertools import izip, chain
from collections import Counter
import json
import cPickle
import os
import hashlib
from keras.preprocessing.sequence import pad_sequences

from IPython import embed

def one_hot(size, index):
    a = numpy.zeros(size, dtype=numpy.bool)
    a[index] = 1
    return a

class Vocab(object):
    '''
        Object with a dictionary of {token: embedding value} pairs.
    '''

    invalid_token = "<<INVALID>>"
    start_token   = "<<START>>"
    end_token     = "<<END>>"

    def __init__(self, infiles, reduce_words=True, caching=True, min_count=5):
        '''
            Create KV embed from a tokenized (space-separated) file(s) of sentences.
        '''
        if type(infiles) is str:
            infiles = [infiles] # in case it is a single file

        cache_file = '{}.pkl'.format(hashlib.md5(''.join(infiles)).hexdigest())
        self.sents_file = '{}.txt'.format(os.path.join('/media', 'Data', 'flipvanrijn', 'datasets', 'text', hashlib.md5(''.join(infiles)).hexdigest()))

        if caching and os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.sentences, self.images, self.img_to_sent, \
                self.word_count, self.special_token_count, self.token_count, \
                self.word_to_int, self.int_to_word, self.start, \
                self.end, self.start_1h, self.end_1h = cPickle.load(fid)
        else:
            self.word_counts = {}
            self.sentences = []
            self.images = []
            self.img_to_sent = {}
            for infile in infiles:
                print 'Processing {}...'.format(infile)

                with open(infile) as f:
                    if 'json' in infile:
                        self.load_json(f)
                    else:
                        self.load_txt(f)

            if reduce_words:
                words = [w for w in self.word_counts if self.word_counts[w] >= min_count]
            else:
                words = [w for w in self.word_counts]
            print 'Reduced words from {} to {} with threshold {}'.format(len(self.word_counts), len(words), min_count)
            alltokens = [self.invalid_token, self.start_token, self.end_token] + words

            self.word_count = len(words)
            self.special_token_count = len(alltokens) - len(words)
            self.token_count = len(alltokens)

            self.word_to_int = {word: i for i, word in enumerate(alltokens)}
            self.int_to_word = {i: word for i, word in enumerate(alltokens)}

            self.start = self.word_to_int[self.start_token]
            self.end = self.word_to_int[self.end_token]
            self.start_1h = one_hot(self.token_count, self.start)
            self.end_1h = one_hot(self.token_count, self.start)

            if caching:
                with open(cache_file, 'wb') as fid:
                    cPickle.dump([self.sentences, self.images, self.img_to_sent, \
                                self.word_count, self.special_token_count, self.token_count, \
                                self.word_to_int, self.int_to_word, self.start, \
                                self.end, self.start_1h, self.end_1h], fid, cPickle.HIGHEST_PROTOCOL)

    def load_txt(self, f):
        for line in f:
            tokens = line.split(' ')
            #self.sentences.append([token.lower() for token in tokens])
            for token in tokens:
                word = token.lower()
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

    def load_json(self, f):
        data = json.load(f);

        for image in data['images']:
            self.images.append(image['filename'])
            for sentence in image['sentences']:
                self.sentences.append([token.lower() for token in sentence['tokens']])
                img_idx, sent_idx = (len(self.images) - 1, len(self.sentences) - 1)
                if img_idx in self.img_to_sent:
                    self.img_to_sent[img_idx].append(sent_idx)
                else:
                    self.img_to_sent[img_idx] = [sent_idx]
                for word in sentence['tokens']:
                    word = word.lower()
                    self.word_counts[word] = self.word_counts.get(word, 0) + 1

    def get(self, token):
        '''
            token -> index
        '''
        i = self.word_to_int.get(token, self.word_to_int[self.invalid_token])
        return i

    def get_1h(self, token):
        '''
            token -> one-hot vector
        '''
        i = self.word_to_int.get(token, self.word_to_int[self.invalid_token])
        return one_hot(self.token_count, i)

    def get_index(self, i):
        '''
            matrix index -> token
        '''
        return self.int_to_word[i]

    def clip(self, sentence):
        '''
            Clips sentence at first occurrence of an eol token.
            eol_tokens contains (token, inclusive=bool) pairs.
        '''
        for i, token in enumerate(sentence):
            for eol, inclusive in self.eol_tokens:
                if token == eol:
                    return sentence[:i + int(inclusive)]

        return sentence

    # used to be convert_sentence
    def sentence_to_1h(self, tokens, reverse=False, pad_length=None):
        vectors = [self.get_1h(x) for x in tokens]
        if pad_length:
            vectors += [self.end_1h]*(pad_length-len(vectors))
        if reverse:
            vectors = list(reversed(vectors))
        return vectors

    def sentence_to_ids(self, tokens, reverse=False, pad_length=None):
        vectors = [self.get(x) for x in tokens]
        if pad_length:
            vectors += [self.end]*(pad_length-len(vectors))
        if reverse:
            vectors = list(reversed(vectors))
        return vectors

    # used to be match_sentence
    def vectors_to_sentence(self, vectors, clip=True):
        tokens = [self.int_to_word[numpy.argmax(v)] for v in vectors]
        if clip:
            tokens = self.clip(tokens)
        return tokens

    def get_matrix(self):
        mat = []
        for sent in self.sentences:
            mat.append(self.sentence_to_ids(sent))
        mat = pad_sequences(mat);
        return mat;

    def matchN(self, vector, n):
        '''
            Takes a probability vector of length token_count and returns the top n tokens.
        '''
        if len(vec) != self.token_count:
            raise Exception("matchN: invalid input: expected vector of length {0} (received length {1})".format(self.token_count, len(vec)))

        matches = [(self.int_to_word[i], i, prob) for i, prob in sorted(enumerate(vector), key=lambda (i, prob): prob, reverse=True)[:n]]
        return tokens

    def convert_sentence(self, *args, **kwargs):
        raise NotImplementedError("convert_sentence: not implemented")

    def match_sentence(self, *args, **kwargs):
        raise NotImplementedError("match_sentence: not implemented")

    def topN(self, *args, **kwargs):
        raise NotImplementedError("topN: not implemented")

    def match1(self, *args, **kwargs):
        raise NotImplementedError("match1: not implemented")

    def eol(self, *args, **kwargs):
        raise NotImplementedError("eol: not implemented")