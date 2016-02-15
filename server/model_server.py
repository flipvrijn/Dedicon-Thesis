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
from preprocessor import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from gensim.models import Word2Vec

from IPython import embed

import sys
sys.path.insert(0, '/home/flipvanrijn/Workspace/Dedicon-Thesis/models/attention/')


class ImageServer(SocketServer.ThreadingTCPServer):
    ''' Generates a caption from an image '''
    def __init__(self, server_address, RequestHandlerClass, args):
        SocketServer.ThreadingTCPServer.__init__(self,server_address, RequestHandlerClass)
        
        self.options_file = args.options
        self.model        = args.model

        print 'Loading options...'
        self._load_options()
        self._load_capgen()
        if 'tex_dim' in self.options:
            print 'Loading text preprocessor...'
            self._load_text_preprocessor(args)

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
        6: loading preprocessor
        '''
        json.dump({'status': status}, open('{}_runningstatus.json'.format(self.model), 'w'))

    def _load_options(self):
        ''' Loads the options of the caption model '''
        self._update_status(1)

        with open(self.options_file, 'rb') as f:
            self.options = pkl.load(f)

    def _load_capgen(self):
        ''' Loads the caption generator version dependent on whether to use textual context or not '''
        if 'tex_dim' in self.options:
            import capgen_text as capgen
            self.capgen = capgen
        else:
            import capgen
            self.capgen = capgen

    def _load_worddict(self):
        ''' Load the worddict '''
        self._update_status(2)

        self.dictionary = dict()
        for kk, vv in self.options['dictionary'].iteritems():
            self.dictionary[vv] = kk
        self.dictionary[0] = '<eos>'
        self.dictionary[1] = 'UNK'

    def _load_text_preprocessor(self, args):
        ''' Load the preprocessor for the context '''
        self._update_status(6)
        self.text_preprocessor = Preprocessor()

        # Load preprocessors based on type of model
        ## TF-IDF:
        if 'tfidf' in self.options['preproc_type']:
            print 'Loading TF-IDF model...'
            with open(args.tfidfmodel, 'rb') as f_tfidf, open(args.svdmodel, 'rb') as f_svd:
                tfidf_model = pkl.load(f_tfidf)
                svd_model   = pkl.load(f_svd) if 'with_svd' in self.options['preproc_params'] else None
                self.text_preprocessor.set_tfidf(tfidf_model, svd_model)
        ## Word2Vec:
        if 'w2v' in self.options['preproc_type']:
            print 'Loading Word2Vec model...'
            w2v_model = Word2Vec.load_word2vec_format(args.w2vmodel, binary=True)
            self.text_preprocessor.set_w2v(w2v_model)
        ## Raw:
        if 'raw' in self.options['preproc_type']:
            print 'Loading counter model...'
            with open(args.rawmodel, 'rb') as f_raw:
                raw_model = pkl.load(args.rawmodel)
                self.text_preprocessor.set_raw(raw_model)

    def resize_image(self, image, resize=256, crop=224):
        ''' Resizes and crops the  image '''
        # From author Kelvin Xu:
        # https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
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
        image_resized = image_resized.convert('RGB')
        data = np.array(image_resized.getdata()).reshape(crop, crop, 3)

        return (image_resized, data)

    def _load_cnn(self):
        ''' Loads the CNN for image encoding '''
        self._update_status(3)

        caffe.set_mode_gpu()

        self._cnn = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
        self._cnn.blobs['data'].reshape(1, 3, 224, 224)

    def _build(self):
        ''' Builds the caption functions '''
        self._update_status(4)

        # build the sampling functions and model
        self.trng    = RandomStreams(1234)
        use_noise    = theano.shared(np.float32(0.), name='use_noise')

        params       = self.capgen.init_params(self.options)
        params       = self.capgen.load_params(self.model, params)
        self.tparams = self.capgen.init_tparams(params)

        # word index
        self.f_init, self.f_next                = self.capgen.build_sampler(self.tparams, self.options, use_noise, self.trng)

        if 'tex_dim' in self.options:
            self.trng, use_noise, inps, \
            alphas, alphas_samples, taus, taus_sample, cost, opt_outs  = self.capgen.build_model(self.tparams, self.options)
        else:
            # get the alphas and selector value [called \beta in the paper]
            self.trng, use_noise, inps, \
            alphas, alphas_samples, cost, opt_outs  = self.capgen.build_model(self.tparams, self.options)

        # create update rules for the stochastic attention
        hard_attn_updates = []
        if self.options['attn_type'] == 'stochastic':
            baseline_time = theano.shared(np.float32(0.), name='baseline_time')
            hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
            hard_attn_updates += opt_outs['attn_updates']
            
        self.f_alpha = theano.function(inps, alphas, name='f_alpha', updates=hard_attn_updates)
        if self.options['selector']:
            self.f_sels = theano.function(inps, opt_outs['selector'], name='f_sels', updates=hard_attn_updates)

        if 'tex_dim' in self.options:
            self.f_tau = theano.function(inps, taus, name='f_tau', updates=hard_attn_updates)
            if self.options['selector']:
                self.f_selts = theano.function(inps, opt_outs['selectort'], name='f_selts', updates=hard_attn_updates)

    def preprocess_img(self, img):
        ''' Preprocess image '''
        img  = img.copy()

        data = img.transpose((2, 0, 1))
        mean = np.asarray([103.939, 116.779, 123.68])
        mean = mean[:, np.newaxis, np.newaxis]
        data -= mean
        data = data[(2, 1, 0), :, :]

        return data

    def preprocess_text(self, text):
        ''' Preprocess text '''
        # Shady part: Dynamically get method based on options (do_tfidf/do_w2v/do_w2vtfidf/do_raw)
        transformer = getattr(self.text_preprocessor, 'do_{}'.format(self.options['preproc_type']))
        # then dynamically plug the parameters into the method
        return transformer(text, **self.options['preproc_params']).astype(np.float32)

    def forward_img(self, data):
        ''' Forward image into the CNN to get encoded image '''
        cnn_in = np.zeros((1, 3, 224, 224), dtype=np.float32)
        cnn_in[0, :] = data

        out = self._cnn.forward_all(blobs=['conv5_4'], **{'data': cnn_in})
        img_context = self._cnn.blobs['conv5_4'].data
        img_context = img_context.transpose((0, 2, 3, 1))

        return img_context[0].reshape([196, 512])

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
        '''
        Expects: pickled({
            img: {
                pixels: bytestring,
                mode: string 'RGB',
                size: tuple,
            },
            text: string,
            introspect_image: bool,
            introspect_context: bool,
            output_path: string,
            num_captions: integer,
        })

        Returns: pickled({
            captions: list,
            with_text_context: bool,
            ?tau: numpy array

        })
        '''
        data = self.recv_msg() # raw pickled data

        unpkl               = pkl.loads(data) # raw unpicled input data
        introspect_image    = unpkl['introspect_image'] # whether to generate the alpha images for introspection
        introspect_context  = unpkl['introspect_context'] # whether to return tau for context introspection
        img                 = Image.frombytes(unpkl['img']['mode'], unpkl['img']['size'], unpkl['img']['pixels'])
        image_resized, data = self.server.resize_image(img)
        img_preprocessed    = self.server.preprocess_img(data) # preprocess image with CNN
        img_context         = self.server.forward_img(img_preprocessed) # image context of the model
        num_captions        = unpkl['num_captions'] if 'num_captions' in unpkl.keys() else 1
        with_text_context   = 'tex_dim' in self.server.options

        if with_text_context:
            text_context = self.server.preprocess_text(unpkl['text'])

            samples, score = self.server.capgen.gen_sample(
                self.server.tparams, self.server.f_init, self.server.f_next, img_context, text_context, 
                self.server.options, trng=self.server.trng, k=num_captions, maxlen=200, stochastic=False)
        else:
            samples, score = self.server.capgen.gen_sample(
                self.server.tparams, self.server.f_init, self.server.f_next, img_context, 
                self.server.options, trng=self.server.trng, k=num_captions, maxlen=200, stochastic=False)

        captions = []
        for sample in samples:
            caption = sample[:-1]

            words = map(lambda w: self.server.dictionary[w] if w in self.server.dictionary.keys() else '<UNK>', caption)
            captions.append(' '.join(words))

        ret = {
            'captions': captions,
            'with_text_context': with_text_context,
        }

        if introspect_context or introspect_image:
            if with_text_context:
                if introspect_image:
                    # Generate the alpha images, e.g. what the model 'sees'
                    alpha = self.server.f_alpha(
                        np.array(caption).reshape(len(caption),1), 
                        np.ones((len(caption),1), dtype='float32'), 
                        img_context.reshape(1,img_context.shape[0],img_context.shape[1]),
                        text_context.reshape(1,text_context.shape[0],text_context.shape[1]))
                if introspect_context:
                    # Generate the tau scores
                    tau = self.server.f_tau(
                        np.array(caption).reshape(len(caption), 1),
                        np.ones((len(caption),1),dtype='float32'),
                        img_context.reshape(1,img_context.shape[0],img_context.shape[1]),
                        text_context.reshape(1,text_context.shape[0],text_context.shape[1]))
                    ret['tau'] = tau
            else:
                if introspect_image:
                    output_path         = unpkl['output_path'].rstrip('/') # output path for introspect images

                    alpha = self.server.f_alpha(
                        np.array(caption).reshape(len(caption),1), 
                        np.ones((len(caption),1), dtype='float32'), 
                        img_context.reshape(1,img_context.shape[0],img_context.shape[1]))

                    if self.server.options['selector']:
                        if with_text_context:
                            sels = self.server.f_sels(np.array(caption).reshape(len(caption),1), 
                                    np.ones((len(caption),1), dtype='float32'), 
                                    img_context.reshape(1,img_context.shape[0],img_context.shape[1]),
                                    text_context.reshape(1,text_context.shape[0],text_context.shape[1]))
                        else:
                            sels = self.server.f_sels(np.array(caption).reshape(len(caption),1), 
                                    np.ones((len(caption),1), dtype='float32'), 
                                    img_context.reshape(1,img_context.shape[0],img_context.shape[1]))

                    out_directory, out_filename = os.path.split(output_path)
                    plt.subplot()
                    plt.imshow(image_resized)
                    plt.axis('off')
                    plt.savefig('{}/{}'.format(out_directory, out_filename))
                            
                    for i, data in enumerate(zip(words, sels)):
                        word, score = data
                        if self.server.options['selector']:
                            word += '(%0.2f)' % score
                        
                            # Upscale alpha weights to 224 x 224
                            alpha_img = skimage.transform.pyramid_expand(alpha[i,0,:].reshape(14,14), upscale=16, sigma=20)
                            plt.subplot()
                            plt.imshow(image_resized)
                            plt.imshow(alpha_img, alpha=0.8)
                            plt.axis('off')
                            plt.set_cmap(cm.Greys_r)
                            plt.savefig('{}/{}_{}'.format(out_directory, i, out_filename))

        # send back the description
        ret = pkl.dumps(ret)
        ret_packed = struct.pack('>I', len(ret)) + ret
        self.request.sendall(ret_packed)

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
    parser.add_argument('--w2vmodel', dest='w2vmodel', help='word2vec model file', type=str)
    parser.add_argument('--tfidfmodel', dest='tfidfmodel', help='tfidf model file', type=str)
    parser.add_argument('--svdmodel', dest='svdmodel', help='svd model file', type=str)
    parser.add_argument('--rawmodel', dest='rawmodel', help='raw model file', type=str)
    args = parser.parse_args()

    main(args)