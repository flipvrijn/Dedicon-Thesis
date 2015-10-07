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

# You will need to add this to make things work
base_path='/media/Data/flipvanrijn/datasets/coco/'
split_path='/media/Data/flipvanrijn/datasets/coco/splits/'

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def main():
    # load file names and captions
    def _load_caps(fname):
        fdict = dict()
        ifdict = dict()
        captions = dict()
        with open(base_path+'annotations/'+fname, 'r') as f:
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

    with open('dictionary.pkl', 'wb') as f:
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
    with open(base_path+'images/train2014list.txt', 'r') as f: #    ls train > train2014list.txt
        for line in f:
            line = re.sub(r'\/.*\/','',line).strip()
            featdict[line] = idx
            idx += 1
    with open(base_path+'images/val2014list.txt', 'r') as f: #    ls val > val2014list.txt
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
    
    def _build_data(flist):
        data_img = [None] * len(flist)
        data_cap = []
        bar = Bar('Processing...', max=len(flist))
        for idx, fname in enumerate(flist):
            # save a sparse matrix
            feature = features_sp[featdict[fname],:]
            data_img[idx] = feature
            for cc in captions[fname]:
                data_cap.append((cc, idx))
            bar.next()
        bar.finish()

        return data_cap, data_img

    print 'Processing Train...'
    data_cap, data_img = _build_data(train_f)
    with open('coco_align.train.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'

    print 'Processing Test...'
    data_cap, data_img = _build_data(test_f)
    with open('coco_align.test.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'

    print 'Processing Dev...'
    data_cap, data_img = _build_data(dev_f)
    with open('coco_align.dev.pkl', 'wb') as f:
        pkl.dump(data_cap, f)
        pkl.dump(data_img, f)
    print 'Done'

if __name__ == '__main__':
    main()
