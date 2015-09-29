import json
import h5py
import argparse
from progress.bar import Bar
from IPython import embed
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='in_file', type=str, help='Path to the input file.')
    parser.add_argument('-o', dest='out_file', type=str, help='Path to the output file.')
    parser.add_argument('--count', dest='min_count', type=int, default=5, help='Minimum number of occurrences for words.')
    args = parser.parse_args()

    print 'Loading JSON file %s...' % args.in_file
    f_in = json.load(open(args.in_file))

    f_out = h5py.File(args.out_file, 'r+')
    num_images = len(f_in['images'])
    tokens = {}
    for i in range(f_out['sentences/tokens'].shape[0]):
        tokens[f_out['sentences/tokens'][i]] = i

    token_count = {}
    max_sentence_size = 0

    # Count the tokens of a sentence
    bar = Bar('Counting', max=num_images, suffix='%(percent)d%%')
    for image_idx, image in enumerate(f_in['images']):

        for s_idx, s in enumerate(image['sentences']):
            max_sentence_size = max_sentence_size if len(s['tokens']) < max_sentence_size else len(s['tokens'])

            for token in s['tokens']:
                token = token.lower()
                token_count[token] = token_count.get(token, 0) + 1
        bar.next()
    bar.finish()

    max_sentence_size += 2 # For adding the #start# and #stop# tokens

    # Remove almost-never-used words
    tokens = [t for t in token_count if token_count[t] >= args.min_count]
    alltokens = ['##invalid##', '##start##', '##stop##'] + tokens

    word_to_int = {word: i for i, word in enumerate(alltokens)}
    int_to_word = {i: word for i, word in enumerate(alltokens)}

    # Save the new tokens
    del f_out['sentences/tokens']
    dtokens = f_out['sentences'].create_dataset('tokens', (len(alltokens),), dtype=h5py.special_dtype(vlen=unicode))
    dtokens[...] = alltokens

    del f_out['sentences/token_ids']
    dtoken_ids = f_out['sentences'].create_dataset('token_ids', (num_images, 5, max_sentence_size), dtype=np.int32)

    bar = Bar('Converting', max=num_images, suffix='%(percent)d%%')
    for image_idx, image in enumerate(f_in['images']):

        for s_idx, s in enumerate(image['sentences']):
            if s_idx > 4:
                break

            ids = np.zeros((max_sentence_size,), dtype=np.int32)
            ids[0] = word_to_int['##start##']
            for t_idx, token in enumerate(s['tokens']):
                token = token.lower()
                ids[t_idx + 1] = word_to_int.get(token, 0)
            ids[len(s['tokens']) + 1] = word_to_int['##stop##']
            dtoken_ids[image_idx,s_idx,:] = ids
        bar.next()
    bar.finish()