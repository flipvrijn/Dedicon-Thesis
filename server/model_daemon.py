import SocketServer
import argparse
import struct

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

from IPython import embed

class ImageServer(SocketServer.ThreadingTCPServer):
    def __init__(self, server_address, RequestHandlerClass, model, options_file):
        SocketServer.ThreadingTCPServer.__init__(self,server_address, RequestHandlerClass)

        self.datasets = {
            'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            'coco': (coco.load_data, coco.prepare_data)
        }
        
        self.options_file = options_file
        self.model        = model

        print 'Loading options...'
        self._load_options()
        print 'Loading data...'
        self._load_data()
        print 'Loading model...'
        self._build()
        print 'All done!'

    def _load_options(self):
        with open(self.options_file, 'rb') as f:
            self.options = pkl.load(f)

    def _load_data(self):
        load_data, prepare_data = self.datasets['coco']

        _, _, _, self.worddict = load_data(False, True, False)

        self.word_idict = dict()
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

    def _build(self):
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
        # context of the model
        msg = self.recv_msg()
        context = pkl.loads(msg)

        use_gt = False # set to False if you want to use the generated sample
        if not use_gt:
            sample, score = capgen.gen_sample(self.server.tparams, self.server.f_init, self.server.f_next, context, 
                                              self.server.options, trng=self.server.trng, k=1, maxlen=200, stochastic=False)
            sidx = numpy.argmin(score)
            caption = sample[sidx][:-1]

        if use_gt:
            caption = map(lambda w: self.server.worddict[w] if self.server.worddict[w] < self.server.options['n_words'] else 1, gt.split())
        words = map(lambda w: self.server.word_idict[w] if w in self.server.word_idict else '<UNK>', caption)

        # just send back the same data, but upper-cased
        self.request.sendall(' '.join(words))

def main(args):
    server  = ImageServer((args.host, args.port), ImageHandler, args.model, args.options_file)
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for the MSCOCO dataset')

    parser.add_argument('--model', dest='model', help='Model file', type=str)
    parser.add_argument('--options', dest='options_file', help='Options file', type=str)
    parser.add_argument('--host', dest='host', help='Host', default='localhost', type=str)
    parser.add_argument('-p', dest='port', help='Port', default=9999, type=int)

    args = parser.parse_args()

    main(args)