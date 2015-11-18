import numpy as np
from scipy.sparse import csr_matrix
from itertools import compress
import json
import cPickle

from progress.bar import Bar
from nltk.tokenize import wordpunct_tokenize

from IPython import embed

def save_sparse_matrix(output_file, matrix):
    out = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
    }
    np.savez(output_file, **out)

def reduce_split(file_name, all_reduced_images_list):
    split = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip() in all_reduced_images_list:
                split.append(line.strip())

    return split

def save_split(file_name, lst):
    with open(file_name, 'w') as f:
        for l in lst:
            print>>f, l

def main():
    split               = 'val'
    contexts_file       = '/media/Data_/flipvanrijn/datasets/coco/processed/full/'+split+'_context.npz'
    images_file         = '/media/Data_/flipvanrijn/datasets/coco/processed/full/'+split+'2014list.txt'
    all_images_file     = ['/media/Data_/flipvanrijn/datasets/coco/processed/reduced/train2014list.txt', '/media/Data/flipvanrijn/datasets/coco/processed/reduced/val2014list.txt']
    captions_file       = '/media/Data_/flipvanrijn/datasets/coco/processed/full/captions_'+split+'2014.json'
    raw_context_file    = '/media/Data_/flipvanrijn/datasets/coco/processed/full/coco_'+split+'_context.pkl'
    image_feats         = '/media/Data_/flipvanrijn/datasets/coco/processed/full/features_'+split+'_conv5_4.npz'
    output_dir          = '/media/Data_/flipvanrijn/datasets/coco/processed/reduced'
    splits_dir          = '/media/Data_/flipvanrijn/datasets/coco/processed/full/splits'

    # Enable / Disable reductions
    flags = {
        'context': True,
        'images_list': False,
        'captions': False,
        'raw_context': False,
        'image_feats': False,
        'splits': False,
    }

    # Loading context info
    context_loader = np.load(contexts_file)
    context_mask = context_loader['mask'].astype(bool)

    # for each split
    # step 1: reduce the context sparse matrix
    if flags['context']:
        print 'Reducing context...'
        context_matrix = csr_matrix((context_loader['data'], context_loader['indices'], context_loader['indptr']), shape=context_loader['shape'])
        reduced_context_matrix = context_matrix[context_mask, :]

    # step 2: reduce image list
    if flags['images_list']:
        print 'Reducing images list...'
        with open(images_file, 'r') as f:
            image_list = f.read().split()
        reduced_image_list = list(compress(image_list, context_mask.tolist()))

    # step 3: reduce captions
    if flags['captions']:
        print 'Reducing captions...'
        new_caps = {}
        fdict = dict()
        ifdict = dict()
        with open(captions_file, 'r') as f:
            caps_js = json.load(f)
            # info & licenses
            new_caps['info'] = caps_js['info']
            new_caps['licenses'] = caps_js['licenses']

            # images
            new_caps['images'] = []
            for img in caps_js['images']:
                if img['file_name'] in reduced_image_list:
                    fdict[img['file_name'].strip()] = img['id']
                    ifdict[img['id']] = img['file_name'].strip()
                    new_caps['images'].append(img)

            # captions
            bar = Bar('Captions ...', max=len(caps_js['annotations']))
            new_caps['annotations'] = []
            for cap in caps_js['annotations']:
                if cap['image_id'] in ifdict.keys():
                    new_caps['annotations'].append(cap)
                bar.next()
            bar.finish()
    
    # step 4: reduce raw context
    if flags['raw_context']:
        print 'Reducing raw context...'
        with open(raw_context_file, 'r') as f:
            titles = cPickle.load(f)
            descs  = cPickle.load(f)
            tags   = cPickle.load(f)

        reduced_titles = list(compress(titles, context_mask.tolist()))
        reduced_descs  = list(compress(descs, context_mask.tolist()))
        reduced_tags   = list(compress(tags, context_mask.tolist()))

    # step 5: reduce image features
    if flags['image_feats']:
        print 'Reducing image features...'
        feats_loader = np.load(image_feats)
        feats_matrix = csr_matrix((feats_loader['data'], feats_loader['indices'], feats_loader['indptr']), shape=feats_loader['shape'])
        reduced_feats_matrix = feats_matrix[context_mask, :]

    # step 6: reduce splits
    if flags['splits']:
        print 'Reducing splits...'
        all_reduced_images_list = []
        for file_name in all_images_file:
            with open(file_name, 'r') as f:
                all_reduced_images_list += f.read().split()
        print '[1/3]'
        train_f = reduce_split('{}/coco_train.txt'.format(splits_dir), all_reduced_images_list)
        print '[2/3]'
        test_f = reduce_split('{}/coco_test.txt'.format(splits_dir), all_reduced_images_list)
        print '[3/3]'
        dev_f = reduce_split('{}/coco_val.txt'.format(splits_dir), all_reduced_images_list)

    # step 7: output everything
    print 'Write to files...'
    if flags['context']:
        print 'Context matrix'
        save_sparse_matrix('{}/{}_context.npz'.format(output_dir, split), reduced_context_matrix)

    if flags['images_list']:
        print 'Images list'
        with open('{}/{}2014list.txt'.format(output_dir, split), 'w') as f:
            for im in reduced_image_list:
                print>>f, im

    if flags['captions']:
        print 'Captions'
        with open('{}/captions_{}2014.json'.format(output_dir, split), 'w') as f:
            json.dump(new_caps, f)

    if flags['raw_context']:
        print 'Raw context'
        with open('{}/coco_{}_context.pkl'.format(output_dir, split), 'w') as f:
            cPickle.dump(reduced_titles, f)
            cPickle.dump(reduced_descs, f)
            cPickle.dump(reduced_tags, f)

    if flags['image_feats']:
        print 'Image features'
        save_sparse_matrix('{}/features_{}_conv5_4.npz'.format(output_dir, split), reduced_feats_matrix)

    if flags['splits']:
        print 'Splits'
        save_split('{}/splits/coco_train.txt'.format(output_dir), train_f)
        save_split('{}/splits/coco_test.txt'.format(output_dir), test_f)
        save_split('{}/splits/coco_val.txt'.format(output_dir), dev_f)

if __name__ == '__main__':
    main()