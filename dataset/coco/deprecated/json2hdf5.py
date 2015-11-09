import json
import h5py
import argparse
import sys
import os

import numpy as np
from progress.bar import Bar
from IPython import embed
from collections import OrderedDict

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', dest='in_file', type=str, help='Path to the input file.')
        parser.add_argument('-o', dest='out_file', type=str, help='Path to the output file.')

        args = parser.parse_args()

        print 'Loading JSON file %s...' % args.in_file
        f_in = json.load(open(args.in_file))

        num_images = len(f_in['images'])

        if (os.path.isfile(args.out_file)):
            f_out = h5py.File(args.out_file, 'r+')
            dtoken_ids  = f_out['sentences/token_ids']
            dnames      = f_out['sentences/image_names']
            ddescs      = f_out['context/descriptions']
            dtitles     = f_out['context/titles']
            dtag_ids    = f_out['context/tags']
            durl        = f_out['context/urls']

            tokens = {}
            for i in range(f_out['sentences/tokens'].shape[0]):
                tokens[f_out['sentences/tokens'][i]] = i
        else:
            f_out    = h5py.File(args.out_file, 'w')
            grp_sents   = f_out.create_group('sentences')
            grp_context = f_out.create_group('context')
            
            # Sentences
            dnames      = grp_sents.create_dataset('image_names', (num_images,), dtype=h5py.special_dtype(vlen=bytes))

            # Context
            ddescs   = grp_context.create_dataset('descriptions', (num_images,), dtype=h5py.special_dtype(vlen=unicode))
            dtitles  = grp_context.create_dataset('titles', (num_images,), dtype=h5py.special_dtype(vlen=unicode))
            dtag_ids = grp_context.create_dataset('tags', (num_images,), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            durl     = grp_context.create_dataset('urls', (num_images,), dtype=h5py.special_dtype(vlen=bytes))

            tokens = {}
            tags   = {}

        bar = Bar('Converting', max=num_images, suffix='%(percent)d%%')
        for image_idx, image in enumerate(f_in['images']):
            dnames[image_idx]   = image['filename']

            for s_idx, s in enumerate(image['sentences']):
                if s_idx > 4:
                    break

                token_ids = []
                for token in s['tokens']:
                    token = token.lower()
                    if token not in tokens.keys():
                        tokens[token] = len(tokens)
                    token_ids.append(tokens[token])
                dtoken_ids[image_idx][s_idx] = token_ids
            embed()

            if image['context']:
                ddescs[image_idx]   = image['context']['description']
                dtitles[image_idx]  = image['context']['title']
                durl[image_idx]     = image['context']['url']

                tag_ids = []
                for tag in image['context']['tags']:
                    tag = tag.lower()
                    if tag not in tags.keys():
                        tags[tag]   = len(tags)
                    tag_ids.append(tags[tag])
                dtag_ids[image_idx]    = tag_ids
            else:
                ddescs[image_idx]   = None
                dtitles[image_idx]  = None
                durl[image_idx]     = None
                dtag_ids[image_idx]    = []

            bar.next()
        bar.finish()

        dtokens = grp_sents.create_dataset('tokens', (len(tokens),), dtype=h5py.special_dtype(vlen=unicode))
        tokens_sorted = OrderedDict(sorted(tokens.items(), key=lambda t: t[1]))
        dtokens[...] = tokens_sorted.keys()

        dtags = grp_sents.create_dataset('tags', (len(tags),), dtype=h5py.special_dtype(vlen=unicode))
        tags_sorted = OrderedDict(sorted(tags.items(), key=lambda t: t[1]))
        dtags[...] = tags_sorted.keys()

        f_out.close()
    except:
        error_info = sys.exc_info()
        print 'Error!', error_info

        embed()