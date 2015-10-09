import SocketServer
import argparse
import struct
import daemon
import caffe
import json
import os.path

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io
import skimage.util

from PIL import Image

import sys
sys.path.insert(0, '/home/flipvanrijn/Workspace/Dedicon-Thesis/networks/arctic-captions/')

import capgen
import generate_caps as gencaps

class ImageServer(SocketServer.ThreadingTCPServer):
    def __init__(self, server_address, RequestHandlerClass, args):
        SocketServer.ThreadingTCPServer.__init__(self,server_address, RequestHandlerClass)
        
        self.options_file = args.options
        self.model        = args.model

        print 'Loading options...'
        self._load_options()
        print 'Loading dictionary...'
        self._load_worddict()
        print 'Loading CNN...'
        self._load_cnn()
        print 'Loading model...'
        self._build()

        self._update_status(5)
        print 'All done!'

    def _update_status(self, status):
        '''
        0: not running
        1: loading_options
        2: loading dictionary
        3: loading CNN
        4: building model
        5: done
        '''
        json.dump({'status': status}, open('{}_runningstatus.json'.format(self.model), 'w'))

    def _load_options(self):
        self._update_status(1)

        with open(self.options_file, 'rb') as f:
            self.options = pkl.load(f)

    def _load_worddict(self):
        self._update_status(2)

        with open('/media/Data/flipvanrijn/datasets/coco/dictionary.pkl', 'rb') as f:
            self.worddict = pkl.load(f)

        self.word_idict = dict()
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

    # From author Kelvin Xu:
    # https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    def load_image(self, filename, resize=256, crop=224):
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
        data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)

        return data

    def _load_cnn(self):
        self._update_status(3)

        caffe.set_mode_gpu()

        self._cnn = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
        self._cnn.blobs['data'].reshape(1, 3, 224, 224)

    def _build(self):
        self._update_status(4)

        # build the sampling functions and model
        self.trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.), name='use_noise')

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
            baseline_time = theano.shared(np.float32(0.), name='baseline_time')
            hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
            hard_attn_updates += opt_outs['attn_updates']
            
        self.f_alpha = theano.function(inps, alphas, name='f_alpha', updates=hard_attn_updates)
        if self.options['selector']:
            self.f_sels = theano.function(inps, opt_outs['selector'], name='f_sels', updates=hard_attn_updates)

    def preprocess(self, img):
        img  = img.copy()

        data = img.transpose((2, 0, 1))
        mean = np.asarray([103.939, 116.779, 123.68])
        mean = mean[:, np.newaxis, np.newaxis]
        data -= mean
        data = data[(2, 1, 0), :, :]

        out = self._cnn.forward_all(blobs=['conv5_4'], **{'data': data})
        context = self._cnn.blobs['conv5_4'].data
        context = context.transpose((0, 2, 3, 1))

        return context.reshape([14*14, 512])

class ImageHandler(SocketServer.BaseRequestHandler):

    def recvall(self, n):
        data = ''
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self):
        raw_length = self.recvall(4)
        if not raw_length:
            return None
        length = struct.unpack('>I', raw_length)[0]
        return self.recvall(length)

    def handle(self):
        data = self.recv_msg()
        file_path, introspect = data.split(';')
        # Whether to generate the alpha images for introspection
        introspect = bool(int(introspect))
        img = self.server.load_image(file_path)
        # Context of the model
        context = self.server.preprocess(img)

        sample, score = capgen.gen_sample(self.server.tparams, self.server.f_init, self.server.f_next, context, 
                                          self.server.options, trng=self.server.trng, k=1, maxlen=200, stochastic=False)
        sidx = np.argmin(score)
        caption = sample[sidx][:-1]

        words = map(lambda w: self.server.word_idict[w] if w in self.server.word_idict else '<UNK>', caption)

        if introspect:
            # Generate the alpha images, e.g. what the model 'sees'
            alpha = self.server.f_alpha(np.array(caption).reshape(len(caption),1), 
                np.ones((len(caption),1), dtype='float32'), 
                context.reshape(1,context.shape[0],context.shape[1]))
            if self.server.options['selector']:
                sels = self.server.f_sels(np.array(caption).reshape(len(caption),1), 
                        np.ones((len(caption),1), dtype='float32'), 
                        context.reshape(1,context.shape[0],context.shape[1]))

            filename = os.path.split(file_path)[1]
            img = img.astype('uint8')
            plt.subplot()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('static/images/{}'.format(filename))
                    
            for i, data in enumerate(zip(words, sels)):
                word, score = data
                if self.server.options['selector']:
                    word += '(%0.2f)' % score
                
                    # Upscale alpha weights to 224 x 224
                    alpha_img = skimage.transform.pyramid_expand(alpha[i,0,:].reshape(14,14), upscale=16, sigma=20)
                    plt.subplot()
                    plt.imshow(img)
                    plt.imshow(alpha_img, alpha=0.8)
                    plt.axis('off')
                    plt.set_cmap(cm.Greys_r)
                    plt.savefig('static/images/{}_{}'.format(i, filename))

        # send back the description
        self.request.sendall(' '.join(words))

def main(args):
    server  = ImageServer((args.host, args.port), ImageHandler, args)
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts image caption server')

    parser.add_argument('--prototxt', dest='prototxt', help='cnn model', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='cnn caffemodel file', type=str)
    parser.add_argument('--model', dest='model', help='model file', type=str)
    parser.add_argument('--options', dest='options', help='options file', type=str)
    parser.add_argument('--host', dest='host', default='localhost', help='Output directory', type=str)
    parser.add_argument('--port', dest='port', default=9999, type=int, help='port')

    args = parser.parse_args()

    main(args)