import h5py
import argparse
import numpy as np
from progress.bar import Bar

from IPython import embed

def ids_to_1h(ids, shape):
    a = np.zeros(shape, dtype=np.bool_)
    for i in range(ids.shape[0]):
        a[i, ids[i]] = 1
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='in_file', type=str, help='Path to the .h5 file.')
    args = parser.parse_args()

    f_in = h5py.File(args.in_file)

    num_images = f_in['sentences/image_names'].shape[0]
    vocab_size = f_in['sentences/tokens'].shape[0]
    max_sent_size = f_in['sentences/token_ids'].shape[2]

    dtoken_1hs = f_in['sentences'].create_dataset('token_1hs', (num_images, 5, max_sent_size, vocab_size), dtype=np.bool_)

    bar = Bar('One hotting', max=num_images, suffix='%(percent)d%%')
    for img_idx in range(num_images):
        for sent_idx in range(5):
            ids      = f_in['sentences/token_ids'][img_idx][sent_idx]
            stripped = np.trim_zeros(ids)
            dtoken_1hs[img_idx, sent_idx] = ids_to_1h(stripped, (max_sent_size, vocab_size))
        bar.next()
    bar.finish()
