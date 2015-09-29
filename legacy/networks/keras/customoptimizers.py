from keras.utils.theano_utils import shared_zeros, shared_scalar, floatX
from keras.optimizers import Optimizer
from six.moves import zip
import theano.tensor as T

class CustomSGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        super(CustomSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.momentum = shared_scalar(momentum)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            m = shared_zeros(p.get_value().shape)  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_gradients(self, loss, params):

        grads = T.grad(loss, params, disconnected_inputs='warn')

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

        return grads

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "momentum": float(self.momentum.get_value()),
                "decay": float(self.decay.get_value()),
                "nesterov": self.nesterov}