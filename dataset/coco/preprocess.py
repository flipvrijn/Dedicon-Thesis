"""
Preprocess COCO

kelvin.xu@umontreal.ca
"""
import cPickle as pkl
import re
import json
import numpy

from scipy.sparse import csr_matrix, vstack
from nltk.tokenize import wordpunct_tokenize
from collections import OrderedDict
from progress.bar import Bar

from IPython import embed

processing_type = 'reduced' # either 'full'/'reduced'
base_path='/media/Data/flipvanrijn/datasets/coco/processed/'+processing_type+'/'
split_path='/media/Data/flipvanrijn/datasets/coco/processed/'+processing_type+'/splits/'

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def main():
    # load file names and captions
    def _load_caps(fname):
        fdict = dict()
        ifdict = dict()
        captions = dict()
        with open(base_path+fname, 'r') as f:
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

    fdict = dict() # file name -> image id
    ifdict = dict() # reverse dictionary
    captions = dict()

    print 'Loading captions...'
    fd, ifd, caps = _load_caps('captions_train2014.json')
    fdict.update(fd)
    ifdict.update(ifd)
    captions.update(caps)
    fd, ifd, caps = _load_caps('captions_val2014.json')
    fdict.update(fd)
    ifdict.update(ifd)
    captions.update(caps)

    # build dictionary
    wordcount = OrderedDict()
    bar = Bar('Building dicionary...', max=len(captions))
    for kk, vv in captions.iteritems():
        for cc in vv:
            words = cc.split()
            for w in words:
                if w not in wordcount:
                    wordcount[w] = 0
                wordcount[w] += 1
        bar.next()
    bar.finish()
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    with open(base_path+'dictionary.pkl', 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

    # load the splits
    def _load_split(name):
        split = []
        with open(split_path+name, 'r') as f:
            for idx, line in enumerate(f):
                split.append(line.strip())
        return split
                
    print 'Loading splits...'
    train_f = _load_split('coco_train.txt')
    test_f = _load_split('coco_test.txt')
    dev_f = _load_split('coco_val.txt')
    print 'Done'

    # load features
    ## feature list
    featdict = OrderedDict()
    idx = 0
    with open(base_path+'train2014list.txt', 'r') as f: #    ls train > train2014list.txt
        for line in f:
            line = re.sub(r'\/.*\/','',line).strip()
            featdict[line] = idx
            idx += 1
    with open(base_path+'val2014list.txt', 'r') as f: #    ls val > val2014list.txt
        for line in f:
            line = re.sub(r'\/.*\/','',line).strip()
            featdict[line] = idx
            idx += 1

    ## final feature map
    print 'Loading train features...'
    features_sp_train = load_sparse_csr(base_path+'features_train_conv5_4.npz')
    print 'Loading validation features...'
    features_sp_val = load_sparse_csr(base_path+'features_val_conv5_4.npz')
    features_sp = vstack((features_sp_train, features_sp_val), format='csr')
    print 'Done'

    # loads the raw context (title, description, tags)
    #def _load_raw_context(name):
    #    raw_context = []
    #    with open(base_path+name) as f:
    #        titles = pkl.load(f)
    #        descriptions = pkl.load(f)
    #        tags = pkl.load(f)
    #        for i in range(len(titles)):
    #            raw_context.append((titles[i], descriptions[i], tags[i]))
    #    return raw_context

    #print 'Loading raw context...'
    #train_raw_context = _load_raw_context('coco_train_context.pkl')
    #val_raw_context = _load_raw_context('coco_val_context.pkl')
    #raw_context = train_raw_context + val_raw_context
    #print 'Done'

    ## final context feature map
    #print 'Loading train context features...'
    #features_ctx_train = load_sparse_csr(base_path+'train_context.npz')
    #print 'Loading validation context features...'
    #features_ctx_val = load_sparse_csr(base_path+'val_context.npz')
    #features_ctx = vstack((features_ctx_train, features_ctx_val), format='csr')
    
    #print 'Loading one-hot context features...'
    #features_ctx = load_sparse_csr(base_path+'context_onehot.npz')

    #print 'Loading TFIDF context features...'
    #loader = numpy.load(base_path+'tfidf_context_wo_stemming.npz')
    #features_ctx = loader['data']
    #print 'Done'

    print 'Loading TF-IDF context features...'
    features_ctx = load_sparse_csr(base_path+'tfidf_context_wo_lsa.npz')

    def _build_data(flist):
        data_img = [None] * len(flist)
        data_cap = []
        data_ctx = [None] * len(flist)
        #data_ctx_raw = [None] * len(flist)
        bar = Bar('Processing...', max=len(flist))
        for idx, fname in enumerate(flist):
            # save a sparse matrix
            data_img[idx] = features_sp[featdict[fname],:]
            feat_ctx = features_ctx[featdict[fname],:]
            data_ctx[idx] = feat_ctx[numpy.newaxis] if feat_ctx.ndim == 1 else feat_ctx
            #data_ctx_raw[idx] = raw_context[featdict[fname]]
            for cc in captions[fname]:
                data_cap.append((cc, idx))
            bar.next()
        bar.finish()

        return data_cap, data_img, data_ctx#, data_ctx_raw

    print 'Processing Train...'
    data_cap, data_img, data_ctx = _build_data(train_f) #, data_ctx_raw
    with open(base_path+'coco_align.train.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
        pkl.dump(data_ctx, f)
        #pkl.dump(data_ctx_raw, f)
    print 'Done'

    print 'Processing Test...'
    data_cap, data_img, data_ctx = _build_data(test_f) #, data_ctx_raw
    with open(base_path+'coco_align.test.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
        pkl.dump(data_ctx, f)
        #pkl.dump(data_ctx_raw, f)
    print 'Done'

    print 'Processing Dev...'
    data_cap, data_img, data_ctx = _build_data(dev_f) #, data_ctx_raw
    with open(base_path+'coco_align.dev.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
        pkl.dump(data_ctx, f)
        #pkl.dump(data_ctx_raw, f)
    print 'Done'

if __name__ == '__main__':
    main()
