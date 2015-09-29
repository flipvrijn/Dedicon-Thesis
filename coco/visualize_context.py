import h5py
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
import textwrap
from random import shuffle

from IPython import embed

def press(event):
    keys = ['q', 'z', '/']
    if event.key not in keys:
        print 'Unknown key'
    else:
        if event.key == 'q':
            print 'Quit'
        if event.key == 'z':
            dvalidcontext[i] = 1
            print 'Yes'
        elif event.key == '/':
            dvalidcontext[i] = -1
            print 'No'
        plt.close()

if __name__ == '__main__':
    file_in = '/media/Data/flipvanrijn/datasets/coco/sents_train.h5'
    img_dir = '/media/Data/flipvanrijn/datasets/coco/images/train'

    f = h5py.File(file_in, 'r+')
    num_images = f['context/titles'].shape[0]

    if 'valid' not in f['context'].keys():
        dvalidcontext = f['context'].create_dataset('valid', (num_images,), dtype=np.int8)
    else:
        dvalidcontext = f['context/valid']

    indices = np.where(dvalidcontext.value == 0)[0]#list(range(num_images))
    shuffle(indices)

    embed()

    for i in indices:
        title = f['context/titles'][i]
        description = f['context/descriptions'][i]
        img = f['sentences/image_names'][i]

        if img and title:
            fig, ax = plt.subplots(figsize=(12, 12))
            fig.canvas.mpl_connect('key_release_event', press)
            ax.imshow(skimage.io.imread('{}/{}'.format(img_dir, img)), aspect='equal')
            ax.set_title(u'{}'.format(title))
            fig.text(.05, .05, '\n'.join(textwrap.wrap(description, 130)[:5]), backgroundcolor='white')


            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            dvalidcontext[i] = 0