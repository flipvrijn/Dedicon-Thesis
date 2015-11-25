import cPickle
import matplotlib.pyplot as plt
import numpy as np
import gensim
import pandas as pd
import argparse
import json
import time
import os.path
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix, vstack
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from itertools import islice
from collections import OrderedDict
from gensim.models import Word2Vec

from progress.bar import Bar
np.random.seed(1234)

from IPython import embed

stop = stopwords.words('english')
stemmer = SnowballStemmer('english')

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) < n:
        yield result
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def preprocess(s, model):
    global stop, stemmer
    
    ns = []
    for w in s:
        if w and w in model:# and w not in stop:
            w = stemmer.stem(w)
            ns.append(w)

    return ns

def main(args):
    # Load data
    print 'Loading context data...'
    # Read context file: *.pkl
    with open(args.in_file, 'r') as f:
        titles = cPickle.load(f)
        descriptions = cPickle.load(f)
        tags = cPickle.load(f)

    # Load Word2Vec model
    print 'Loading word2vec model...'
    t_model_start = time.time()
    model = Word2Vec.load_word2vec_format('/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin', binary=True)
    print 'Loaded in {}s'.format(time.time() - t_model_start)

    # Create feature vectors of context and only keep images WITH context
    mask = []
    bar = Bar('Extracting features...', max=len(titles))
    context_filtered = []
    lengths = []
    for i in xrange(len(titles)):
        # Stem words and remove stopwords for title...
        context = []
        title = preprocess(titles[i].split(' '), model)
        if title:
            context.append(title)
        # ... description (for each sentence) ...
        for desc in sent_tokenize(descriptions[i]):
            desc = preprocess(desc.split(' '), model)
            if desc:
                context.append(desc)
        # ... and tagsc
        ts = preprocess(tags[i], model)
        if ts:
            context.append(ts)

        X = []
        if context:
            for sent in context:
                X += [' '.join(x) for x in window(sent, args.w)]
            mask.append(1)
        else:
            mask.append(0)

        context_filtered.append(X)
        lengths.append(len(X))
        bar.next()
    bar.finish()

    embed()

    out = {
        'data': context_filtered,
        'mask': np.array(mask, dtype=bool),
    }

    print 'Saving features to {}'.format(args.out_file)
    np.savez('{}'.format(args.out_file), **out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context data for the MSCOCO dataset')
    parser.add_argument('in_file', help='Context file', type=str)
    parser.add_argument('out_file', help='Output file', type=str)
    parser.add_argument('-w', help='Sliding window size', type=int, default=3)
    parser.add_argument('--w2v', dest='w2v', help='Word2Vec model', type=str)

    args = parser.parse_args()

    main(args)