import cPickle
import matplotlib.pyplot as plt
import numpy as np
import gensim
import pandas as pd
import argparse
import json
import time
import os.path
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix, vstack
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from itertools import islice
from collections import OrderedDict
from gensim.models import Word2Vec

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from progress.bar import Bar
np.random.seed(1234)

from IPython import embed

stop = stopwords.words('english')
stemmer = SnowballStemmer('english')

def pipeline_w2v(titles, descriptions, tags):
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

    def encode(model, stem2word, X):
        feat = None
        for i, ngram in enumerate(X):
            words = ngram.split(' ')
            vec = np.array([model[stem2word[w]] for w in words]).prod(axis=0)
            if i == 0:
                feat = vec
            else:
                feat = np.vstack((feat, vec))

        return feat

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

    print 'Building stemming dictionary...'
    stem2word = {}
    t_stemming_start = time.time()
    for word in model.index2word:
        stem = stemmer.stem(word)
        stem2word[stem] = word
    print 'Built in {}s'.format(time.time() - t_stemming_start)

    bar = Bar('Encoding features using Word2Vec...', max=len(context_filtered))
    for i, X in enumerate(context_filtered):
        features = np.zeros((150, 512), dtype=np.float32)
        vec = encode(model, stem2word, X)
        if vec is not None:
            # Use the first N
            vec = vec[:150]
            # Fix the dimensionality of the feature vector
            if vec.ndim != 2:
                vec = vec[np.newaxis, :]
            features[:vec.shape[0], :vec.shape[1]] = vec

        if i == 0:
            feat_flatten = csr_matrix(features.flatten())
        else:
            feat_flatten = vstack([feat_flatten, csr_matrix(features.flatten())])
        bar.next()
    bar.finish()

    return feat_flatten

def pipeline_tfidf(titles, descriptions, tags, dataset_sizes):
    def preprocess(s):
        global stop, stemmer
        
        ns = []
        for w in s:
            if w:
                w = stemmer.stem(w)
                ns.append(w)

        return ns

    docs = []
    bar = Bar('Extracting features...', max=len(titles))
    for i in xrange(len(titles)):
        # Stem words and remove stopwords for title...
        context = []
        title = preprocess(titles[i].split(' '))
        if title:
            context.append(' '.join(title))
        # ... description (for each sentence) ...
        for desc in sent_tokenize(descriptions[i]):
            desc = preprocess(desc.split(' '))
            if desc:
                context.append(' '.join(desc))
        # ... and tagsc
        ts = preprocess(tags[i])
        if ts:
            context.append(' '.join(ts))

        docs.append(' '.join(context))
        bar.next()
    bar.finish()

    print 'Encoding features with TFIDF'
    tfidf     = TfidfVectorizer(min_df=3, max_df=0.8, strip_accents='unicode',
                            analyzer='word', ngram_range=(1,2),
                            use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
    svd       = TruncatedSVD(512)
    normalize = Normalizer(copy=False)
    lsa       = make_pipeline(tfidf, svd, normalize)
    matrix    = lsa.fit_transform(docs)

    # Split the matrix up according to the dataset files
    return matrix

def pipeline_pos(titles, descriptions, tags):
    def preprocess(inpt):
        return inpt

    # Create feature vectors of context and only keep images WITH context
    bar = Bar('Extracting features...', max=len(titles))
    pos_collection = []
    for i in xrange(len(titles)):
        # Stem words and remove stopwords for title...
        context = []
        title = preprocess(titles[i].split(' '))
        if title:
            context.append(title)
        # ... description (for each sentence) ...
        for desc in sent_tokenize(descriptions[i]):
            desc = preprocess(desc.split(' '))
            if desc:
                context.append(desc)
        # ... and tagsc
        ts = preprocess(tags[i])
        if ts:
            context.append(ts)
        
        pos = nltk.pos_tag_sents(context)
        pos = list(itertools.chain(*pos))
        pos_collection.append(pos)
        bar.next()
    bar.finish()

    return pos_collection

def main(args):
    if args.type == 'w2v':
        if ',' in args.in_file:
            raise 'Only one input file is needed for Word2Vec!'

        print 'Loading context data...'
        with open(args.in_file, 'r') as f:
            titles = cPickle.load(f)
            descriptions = cPickle.load(f)
            tags = cPickle.load(f)

        feat_flatten = pipeline_w2v(titles, descriptions, tags)

        out = {
            'data': feat_flatten.data,
            'indices': feat_flatten.indices,
            'indptr': feat_flatten.indptr,
            'shape': feat_flatten.shape,
        }

        print 'Saving features to {}'.format(args.out_file)
        np.savez(args.out_file, **out)
    elif args.type == 'tfidf':
        if ',' not in args.in_file:
            raise 'All context files are required for TFIDF!'

        dataset_sizes = []
        all_titles = []
        all_descriptions = []
        all_tags = []
        for f_in in args.in_file.split(','):
            print 'Loading context data from {}...'.format(f_in)
            with open(f_in, 'r') as f:
                titles = cPickle.load(f)
                descriptions = cPickle.load(f)
                tags = cPickle.load(f)
            dataset_sizes.append(len(titles))
            all_titles += titles
            all_descriptions += descriptions
            all_tags += tags

        data = pipeline_tfidf(all_titles, all_descriptions, all_tags, dataset_sizes)

        out = {
            'data': data,
        }
        
        print 'Saving context features to {}...'.format(args.out_file)
        np.savez(args.out_file, **out)
    elif args.type == 'pos':
        print 'Loading context data...'
        with open(args.in_file, 'r') as f:
            titles = cPickle.load(f)
            descriptions = cPickle.load(f)
            tags = cPickle.load(f)

        nouns_collection = pipeline_pos(titles, descriptions, tags)

        out = {
            'nouns': nouns_collection,
        }
        np.savez(args.out_file, **out)
    else:
        raise 'Unknown feature type!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context data for the MSCOCO dataset')
    parser.add_argument('in_file', help='Context file', type=str)
    parser.add_argument('out_file', help='Output file', type=str)
    parser.add_argument('--window', '-w', help='Sliding window size', type=int, default=3)
    parser.add_argument('--type', '-t', help='Feature type', choices=['w2v', 'tfidf', 'pos'])

    args = parser.parse_args()

    main(args)