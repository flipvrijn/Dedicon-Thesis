import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io

from PIL import Image

import sys
sys.path.insert(0, '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/')

import capgen
import generate_caps as gencaps
import flickr8k
import flickr30k
import coco

class Visualizer(object):

    def __init__(self, model, options_file, image_path, dev_list):
        self.model = model
        self.options_file = options_file
        self.datasets = {
            'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            'coco': (coco.load_data, coco.prepare_data)
        }
        self.image_path = image_path
        self.dev_list = dev_list
        self.running = True # visualization loop is running

        self._load_options()
        self._build()
        self._load_data()

    def _load_image(self, filename, resize=256, crop=224):
        image = Image.open(filename)
        width, height = image.size

        if width > height:
            width = (width * resize) / height
            height = resize
        else:
            height = (height * resize) / width
            width = resize
        left = (width  - crop) / 2
        top  = (height - crop) / 2
        image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
        data = numpy.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
        data = data.astype('float32') / 255
        return data

    def _load_options(self):
        with open(self.options_file, 'rb') as f:
            self.options = pkl.load(f)

    def _load_data(self):
        self.flist = []
        with open(self.dev_list, 'r') as f:
            for l in f:
                self.flist.append(l.strip())

        load_data, prepare_data = self.datasets[self.options['dataset']]

        _, self.valid, _, self.worddict = load_data(False, True, False)
        print 'Data loaded'

        self.word_idict = dict()
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

    def _build(self):
        print 'Building model...'

        # build the sampling functions and model
        self.trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.), name='use_noise')

        params = capgen.init_params(self.options)
        params = capgen.load_params(self.model, params)
        self.tparams = capgen.init_tparams(params)

        # word index
        self.f_init, self.f_next = capgen.build_sampler(self.tparams, self.options, use_noise, self.trng)

        self.trng, use_noise, inps, \
        alphas, alphas_samples, cost, opt_outs = capgen.build_model(self.tparams, self.options)

        # get the alphas and selector value [called \beta in the paper]

        # create update rules for the stochastic attention
        hard_attn_updates = []
        if self.options['attn_type'] == 'stochastic':
            baseline_time = theano.shared(numpy.float32(0.), name='baseline_time')
            hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
            hard_attn_updates += opt_outs['attn_updates']
            
        self.f_alpha = theano.function(inps, alphas, name='f_alpha', updates=hard_attn_updates)
        if self.options['selector']:
            self.f_sels = theano.function(inps, opt_outs['selector'], name='f_sels', updates=hard_attn_updates)

        print 'Done'

    def _press(self, event):
        print event.key
        if event.key == 'q':
            self.running = False
        plt.close()

    def random_sample(self):
        idx = numpy.random.randint(0, len(self.valid[0])) # random image
        k = 1 # beam width
        use_gt = False # set to False if you want to use the generated sample
        gt = self.valid[0][idx][0] # groundtruth
        context = numpy.array(self.valid[1][self.valid[0][idx][1]].todense()).reshape([14*14, 512]) # annotations
        img = self._load_image(self.image_path+self.flist[self.valid[0][idx][1]])

        if not use_gt:
            sample, score = capgen.gen_sample(self.tparams, self.f_init, self.f_next, context, 
                                              self.options, trng=self.trng, k=k, maxlen=200, stochastic=False)
            sidx = numpy.argmin(score)
            caption = sample[sidx][:-1]

        # print the generated caption and the ground truth
        if use_gt:
            caption = map(lambda w: self.worddict[w] if self.worddict[w] < self.options['n_words'] else 1, gt.split())
        words = map(lambda w: self.word_idict[w] if w in self.word_idict else '<UNK>', caption)
        print 'Sample:', ' '.join(words)
        print 'GT:', gt

        return caption, context, img, words

    def visualize(self):
        while self.running:
            caption, context, img, words = self.random_sample()

            alpha = self.f_alpha(numpy.array(caption).reshape(len(caption),1), 
                    numpy.ones((len(caption),1), dtype='float32'), 
                    context.reshape(1,context.shape[0],context.shape[1]))
            if self.options['selector']:
                sels = self.f_sels(numpy.array(caption).reshape(len(caption),1), 
                        numpy.ones((len(caption),1), dtype='float32'), 
                        context.reshape(1,context.shape[0],context.shape[1]))

            # display the visualization
            n_words = alpha.shape[0] + 1
            w = numpy.round(numpy.sqrt(n_words))
            h = numpy.ceil(numpy.float32(n_words) / w)
                    
            fig, axes = plt.subplots(int(w), int(h), figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
            axes.flat[0].imshow(img)
            plt.axis('off')

            fig.canvas.mpl_connect('key_pressed_event', self._press)

            smooth = True

            for ii, data in enumerate(zip(axes.flat[1:], words, sels)):
                axis, word, score = data
                lab = word
                if self.options['selector']:
                    lab += '(%0.2f)' % score
                axis.text(0, 1, lab, backgroundcolor='white', fontsize=13, color='black' if score > 0.1 else 'red')
                axis.imshow(img)
                if smooth:
                    alpha_img = skimage.transform.pyramid_expand(alpha[ii,0,:].reshape(14,14), upscale=16, sigma=20)
                else:
                    alpha_img = skimage.transform.resize(alpha[ii,0,:].reshape(14,14), [img.shape[0], img.shape[1]])
                axis.imshow(alpha_img, alpha=0.8, cmap=cm.Greys_r)
            plt.show()

def main():
    model       = '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/my_caption_model.npz'
    options     = '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/my_caption_model.npz.pkl'
    image_path  = '/media/Data/flipvanrijn/datasets/coco/images/val/'
    dev_list    = '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/splits/coco_val.txt'
    visualizer = Visualizer(model, options, image_path, dev_list)
    visualizer.visualize()

if __name__ == '__main__':
    main()