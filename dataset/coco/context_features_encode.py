import numpy as np
import argparse
import skipthoughts
import time
from scipy.sparse import csr_matrix, vstack
from gensim.models import Word2Vec

from progress.bar import Bar
from IPython import embed

def encode(model, X):
    feat = None
    for i, ngram in enumerate(X):
        words = ngram.split(' ')
        vec = np.array([model[w.lower()] for w in words]).prod(axis=0)
        if i == 0:
            feat = vec
        else:
            feat = np.vstack((feat, vec))

    return feat

def main(args):
    loader = np.load(args.in_file)
    context = loader['data']

    # Load Word2Vec model
    print 'Loading word2vec model...'
    t_model_start = time.time()
    model = Word2Vec.load_word2vec_format('/media/Data_/flipvanrijn/models/word2vec/enwiki-latest-pages.512.bin', binary=True)
    print 'Loaded in {}s'.format(time.time() - t_model_start)

    bar = Bar('Extracting features...', max=len(context))
    try:
        for i, X in enumerate(context):
            features = np.zeros((100, 512), dtype=np.float32)
            vec = encode(model, X)
            if vec is not None:
                # Use the first N ngrams
                vec = vec[:100]
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
    except:
        embed()
        raise

    out = {
        'data': feat_flatten.data,
        'indices': feat_flatten.indices,
        'indptr': feat_flatten.indptr,
        'shape': feat_flatten.shape,
        'mask': np.array(loader['mask'], dtype=bool),
    }

    print 'Saving features to {}'.format(args.out_file)
    np.savez('{}'.format(args.out_file), **out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate context data for the MSCOCO dataset')
    parser.add_argument('in_file', help='Input file', type=str)
    parser.add_argument('out_file', help='Output file', type=str)
    args = parser.parse_args()

    main(args)