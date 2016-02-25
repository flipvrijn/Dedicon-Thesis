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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from progress.bar import Bar
np.random.seed(1234)

from IPython import embed

stop = stopwords.words('english')
stemmer = SnowballStemmer('english')

def pipeline_w2vtfidf(titles, descriptions, tags, n=1, n_best=150):
    def preprocess(s):
        global stemmer
        
        ns = []
        raw = []
        for w in s:
            if w:
                stemmed = stemmer.stem(w)
                ns.append(stemmed)
                raw.append(w)

        return ns, raw

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
    w2v_model = Word2Vec.load_word2vec_format('/media/Data/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin', binary=True)
    print 'Loaded in {}s'.format(time.time() - t_model_start)

    print 'Loading TF-IDF model...'
    t_tfidf_start = time.time()
    with open('/media/Data/flipvanrijn/datasets/coco/processed/reduced/tfidf_model.pkl') as f_in:
        tfidf_model = cPickle.load(f_in)
    tfidf_feature_names = tfidf_model.get_feature_names()
    print 'Loaded in {}s'.format(time.time() - t_tfidf_start)

    print 'Building stemming dictionary...'
    stem2word = {}
    t_stemming_start = time.time()
    for word in w2v_model.index2word:
        stem = stemmer.stem(word)
        stem2word[stem] = word
    print 'Built in {}s'.format(time.time() - t_stemming_start)

    # Create feature vectors of context
    bar = Bar('Extracting features...', max=len(titles))
    for i in xrange(len(titles)):
        # Stem words and remove stopwords for title...
        context = []
        context_raw = []
        title, title_raw = preprocess(titles[i].split(' '))
        if title:
            context += title
            context_raw += title_raw
        # ... description (for each sentence) ...
        for sent in sent_tokenize(descriptions[i]):
            sent, sent_raw = preprocess(sent.split(' '))
            if sent:
                context += sent
                context_raw += sent_raw
        # ... and tagsc
        ts, ts_raw = preprocess(tags[i])
        if ts:
            context += ts
            context_raw += ts_raw

        # Get TF-IDF count for current context
        feats = tfidf_model.transform([' '.join(context)]).todense()[0].tolist()[0]
        # Sort on score and grab best n_best
        scores = sorted(
            [(tfidf_feature_names[pair[0]], pair[1]) for pair in zip(range(0, len(feats)), feats) if pair[1] > 0], 
            key=lambda x: x[1] * -1
        )[:n_best]

        # Construct W2V feature matrix from the N best words
        features = np.zeros((n_best, 512), dtype=np.float32)
        num_features = 0
        for stem, score in scores:
            # skip n-grams or unknown words
            if ' ' in stem or stem not in stem2word:
                continue
            word = stem2word[stem]
            features[num_features] = w2v_model[word]
            num_features += 1

        if i == 0:
            feat_flatten = csr_matrix(features.flatten())
        else:
            feat_flatten = vstack([feat_flatten, csr_matrix(features.flatten())])
        bar.next()
    bar.finish()

    return feat_flatten

def pipeline_w2v(titles, descriptions, tags, args):
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
            if w and w in model:
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
                X += [' '.join(x) for x in window(sent, args.window)]
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

def pipeline_tfidf(titles, descriptions, tags):
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

    print 'Encoding features with TF-IDF'
    model     = TfidfVectorizer(min_df=3, max_df=0.8, strip_accents='unicode',
                            analyzer='word', ngram_range=(1,2),
                            use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')

    matrix = model.fit_transform(docs)

    return (matrix, model)

def pipeline_lsa(tfidf_matrix):
    print 'Encoding features with LSA'
    
    svd       = TruncatedSVD(512)
    normalize = Normalizer(copy=False)
    lsa       = make_pipeline(svd, normalize)
    lsa_mat   = lsa.fit_transform(tfidf_matrix)

    # Split the matrix up according to the dataset files
    return lsa_mat, lsa

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

def pipeline_onehot(titles, descriptions, tags):
    # Create feature vectors of context and only keep images WITH context
    bar = Bar('Extracting features...', max=len(titles))
    docs = []
    for i in xrange(len(titles)):
        docs.append(u'{} {} {}'.format(titles[i], descriptions[i], ' '.join(tags[i])))

    vectorizer = CountVectorizer(min_df=5)
    X = vectorizer.fit_transform(docs)

    bar = Bar('Extracting features...', max=len(docs))
    idx_docs = []
    for idoc, doc in enumerate(docs):
        idxs    = X[idoc].nonzero()[1] + 1
        idxs    = idxs.tolist()
        idx_docs.append(idxs)
        bar.next()
    bar.finish()

    max_len = 500

    bar = Bar('Merging into one matrix...', max=len(idx_docs))
    for i, idx_doc in enumerate(idx_docs):
        features = np.zeros((1, max_len), np.int64)
        vec = np.array(idx_doc[:max_len])
        features[0, :vec.shape[0]] = vec

        if i == 0:
            feat_flatten = csr_matrix(features.flatten())
        else:
            feat_flatten = vstack([feat_flatten, csr_matrix(features.flatten())])
        bar.next()
    bar.finish()

    return feat_flatten, vectorizer

def main(args):
    if args.type == 'w2v' or args.type == 'w2vtfidf':
        if ',' in args.in_file:
            raise 'Only one input file is needed for Word2Vec!'

        print 'Loading context data...'
        with open(args.in_file, 'r') as f:
            titles = cPickle.load(f)
            descriptions = cPickle.load(f)
            tags = cPickle.load(f)

        if args.type == 'w2v':
            feat_flatten = pipeline_w2v(titles, descriptions, tags, args)
        else:
            feat_flatten = pipeline_w2vtfidf(titles, descriptions, tags)

        out = {
            'data': feat_flatten.data,
            'indices': feat_flatten.indices,
            'indptr': feat_flatten.indptr,
            'shape': feat_flatten.shape,
        }

        print 'Saving features to {}'.format(args.out_file)
        np.savez(args.out_file, **out)
    elif args.type == 'tfidf' or args.type == 'lsa':
        if ',' not in args.in_file:
            raise 'All context files are required for TFIDF/LSA!'

        # Load full dataset
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

        # Generate TF-IDF matrix
        data, tfidf_model = pipeline_tfidf(all_titles, all_descriptions, all_tags)

        if args.type == 'tfidf':
            print 'Saving TF-IDF context features to {}...'.format(args.out_file)
            with open(args.out_file, 'wb') as f_out:
                cPickle.dump(tfidf_model, f_out)           

        # In case of LSA, also apply LSA to the TF-IDF matrix
        if args.type == 'lsa':
            data, lsa = pipeline_lsa(data)

        embed()
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
    elif args.type == 'onehot':
        if ',' not in args.in_file:
            raise 'All context files are required for onehot encoding!'

        all_titles = []
        all_descriptions = []
        all_tags = []
        for f_in in args.in_file.split(','):
            print 'Loading context data from {}...'.format(f_in)
            with open(f_in, 'r') as f:
                titles = cPickle.load(f)
                descriptions = cPickle.load(f)
                tags = cPickle.load(f)
            all_titles += titles
            all_descriptions += descriptions
            all_tags += tags

        feats, vectorizer = pipeline_onehot(all_titles, all_descriptions, all_tags)

        out = {
            'data': feats.data,
            'indices': feats.indices,
            'indptr': feats.indptr,
            'shape': feats.shape,
        }

        print 'Saving features to {}'.format(args.out_file)
        np.savez(args.out_file, **out)
    else:
        raise 'Unknown feature type!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context data for the MSCOCO dataset')
    parser.add_argument('in_file', help='Context file', type=str)
    parser.add_argument('out_file', help='Output file', type=str)
    parser.add_argument('--window', '-w', help='Sliding window size', type=int, default=3)
    parser.add_argument('--type', '-t', help='Feature type', choices=['w2v', 'tfidf', 'lsa', 'pos', 'w2vtfidf', 'onehot'])

    args = parser.parse_args()

    main(args)