from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Tanh, Linear

from theano import tensor
import theano
import numpy as np
import argparse
import hdf5storage

from IPython import embed

class MLP(object):
    def __init__(self, n_in, n_hidden):
        self.x = tensor.matrix('features')
        self.in_to_h = Linear(name='in_to_h', input_dim=n_in, output_dim=n_hidden)
        self.h       = Tanh().apply(self.in_to_h.apply(self.x))

        self.in_to_h.weights_init = IsotropicGaussian(0.1)
        self.in_to_h.biases_init  = Constant(0)
        self.in_to_h.initialize()

        self.f = theano.function([self.x], self.h)

    def activate(self, inpt):
        return self.f(inpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode the image representation')
    parser.add_argument('input', type=str, help='File with RCNN output')
    parser.add_argument('output', type=str, help='File to write MLP output')
    parser.add_argument('--nhidden', default=500, type=int, help='Number of hidden units')

    args = parser.parse_args()

    mlp = MLP(4096, args.nhidden)

    data = hdf5storage.read(path='/', filename=args.input)
    data = data[0][0][0]

    v = np.empty((len(data), args.nhidden))
    for idx, d in enumerate(data):
        v[idx] = mlp.activate(np.array([d]))

    embed()