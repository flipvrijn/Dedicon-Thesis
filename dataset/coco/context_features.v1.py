import cPickle
import matplotlib.pyplot as plt
import numpy as np
import gensim
import pandas as pd
import argparse
import json
import os.path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, vstack
from nltk.tokenize import wordpunct_tokenize

from collections import OrderedDict

from progress.bar import Bar
np.random.seed(1234)

from IPython import embed

stop = stopwords.words('english')
stemmer = WordNetLemmatizer()

def preprocess(s, model):
    global stop, stemmer
    
    ns = []
    for w in s:
        w = stemmer.lemmatize(w)
        if w in model and w not in stop:
            ns.append(w)
        
    return ns

def sim(s1, s2, model):
    s1 = preprocess(s1, model)
    s2 = preprocess(s2, model)
    similarity = model.n_similarity(s1, s2)

    return np.sum(similarity)

def n_relevant(caption, title, description, tags, model, n=10):
    scores = []
    w_caption = caption.split(' ')
    w_title   = title.split(' ')
    if title:
        try:
            scores.append((sim(w_caption, w_title, model), title))
        except:
            print w_title
            raise
    for desc in description.split('.'):
        w_desc = desc.split(' ')
        if desc:
            scores.append((sim(w_caption, w_desc, model), desc))
    if tags:
        scores.append((sim(w_caption, tags, model), ' '.join(tags)))
    
    df = pd.DataFrame(scores, columns=['score', 'sentence'])
    df = df.dropna()
    df = df.sort(['score'], ascending=False)[:n]
    return df['score']

def n_relevant_words(caption, title, description, tags, model, n=100):
    def relevant(s1, s2, model):
        m = np.zeros((len(s1), len(s2)))
        for ii in xrange(len(s1)):
            for jj in xrange(len(s2)):
                if s1[ii] in model and s2[jj] in model:
                    m[ii][jj] = model.similarity(s1[ii], s2[jj])
        scores = []
        for r, c in zip(np.argmax(m, axis=0), range(len(s2))):
            scores.append((m[r][c], s2[c]))
        return scores
    
    cap = preprocess(caption, model)
    title = preprocess(title, model)
    descs = description.split('.')
    ts = preprocess(tags, model)

    scores = []
    scores += relevant(cap, title, model)
    for desc in descs:
        desc = preprocess(desc.split(' '), model)
        scores += relevant(cap, desc, model)
    scores += relevant(cap, ts, model)

    df = pd.DataFrame(scores, columns=['score', 'word'])
    df = df.sort(['score'], ascending=False)
    df = df.dropna()
    df = df.drop_duplicates()
    return df['word'][:n]

# load file names and captions
def _load_caps(fname):
    fdict = dict()
    ifdict = dict()
    captions = dict()
    with open(fname, 'r') as f:
        caps_js = json.load(f)
        # images
        for img in caps_js['images']:
            fdict[img['file_name'].strip()] = img['id']
            ifdict[img['id']] = img['file_name'].strip()

        # captions
        for cap in caps_js['annotations']:
            sent = ' '.join(wordpunct_tokenize(cap['caption'].strip())).lower()
            if ifdict[cap['image_id']] in captions:
                captions[ifdict[cap['image_id']]].append(sent)
            else:
                captions[ifdict[cap['image_id']]] = [sent]

    return fdict, ifdict, captions

def main(args):
    # Load data
    print 'Loading context data...'
    # Read context file: *.pkl
    with open(args.in_file, 'r') as f:
        titles = cPickle.load(f)
        descriptions = cPickle.load(f)
        tags = cPickle.load(f)

    fdict = dict() # file name -> image id
    ifdict = dict() # reverse dictionary
    captions = dict()

    print 'Loading captions...'
    fd, ifd, caps = _load_caps(args.captions_file)
    fdict.update(fd)
    ifdict.update(ifd)
    captions.update(caps)

    o_captions = OrderedDict(sorted(captions.items(), key=lambda x: x[0]))

    # Read images list <split>2014list.txt
    with open(args.img_list, 'r') as f:
        images = f.read().split()

    # Load Word2Vec model
    print 'Loading Word2Vec model...'
    model = gensim.models.Word2Vec.load_word2vec_format(args.w2v, binary=True)

    # Create feature vectors of context and only keep images WITH context
    mask = []
    bar = Bar('Extracting features...', max=len(o_captions))
    raw_words = []
    for i, (img, caps) in enumerate(o_captions.items()):
        cap = np.random.choice(caps)
        title = titles[i].split(' ')
        desc = descriptions[i]
        ts = tags[i]

        # use 100 most relevant words
        relevant_words = n_relevant_words(cap, title, desc, ts, model, n=args.n)
        # save it for reference
        raw_words.append(raw_words)
        
        context = np.zeros((args.n, model.vector_size), dtype=np.float32)
        # check if it has any relevant words
        if len(relevant_words) > args.thresh:
            mask.append(True)

            for wi, w in enumerate(relevant_words.values):
                context[wi,:] = model[w]
        else:
            mask.append(False)

        if i == 0:
            feat_flatten = csr_matrix(context.flatten())
        else:
            feat_flatten = vstack([feat_flatten, csr_matrix(context.flatten())])

        bar.next()
    bar.finish()

    out = {
        'data': feat_flatten.data,
        'indices': feat_flatten.indices,
        'indptr': feat_flatten.indptr,
        'shape': feat_flatten.shape,
        'mask': np.array(mask, dtype=bool),
    }

    print 'Saving features to {}'.format(args.out_file)
    np.savez('{}'.format(args.out_file), **out)

    raw_out_file = args.out_file+'_raw.pkl'
    print 'Saving relevant words to {}'.format(raw_out_file)
    with open(raw_out_file, 'r') as f:
        cPickle.dump(raw_words, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context data for the MSCOCO dataset')
    parser.add_argument('in_file', help='Context file', type=str)
    parser.add_argument('captions_file', help='Captions file', type=str)
    parser.add_argument('img_list', help='File with list of image filenames for the split', type=str)
    parser.add_argument('out_file', help='Output file', type=str)
    parser.add_argument('--w2v', dest='w2v', help='Word2Vec model', type=str)
    parser.add_argument('--n_relevant_w', dest='n', help='N relevant words for features', type=int, default=100)
    parser.add_argument('--thresh', dest='thresh', help='Minimum number of relevant words', type=int, default=0)

    args = parser.parse_args()

    main(args)