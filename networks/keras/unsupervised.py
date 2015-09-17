import theano

import theano.tensor as T

from keras.utils.theano_utils import shared_zeros, floatX
from keras.models import Model, standardize_X
from keras.layers import containers
from keras import optimizers, objectives
from IPython.core.debugger import Tracer

class Unsupervised(Model, containers.Sequential):
    """
    Single layer unsupervised learning Model.
    """
    # compile theano graph, adapted from keras.models.Sequential
    def compile(self, optimizer, loss, theano_mode=None):
        self.optimizer = optimizers.get(optimizer)

        self.loss = objectives.get(loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        self.scores = T.matrix('scores')

        train_loss = self.loss(self.y_train, self.scores)
        test_loss = self.loss(self.y_test, self.scores)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'

        self.theano_mode = theano_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.params, self.constraints, train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins   = self.X_train
            test_ins    = self.X_test
            predict_ins = self.X_test 
        else:
            train_ins   = [self.X_train]
            test_ins    = [self.X_test]
            predict_ins = [self.X_test]

        train_ins += [self.scores]
        test_ins  += [self.scores]

        self._train     = theano.function(train_ins, train_loss, updates=updates,
                                     allow_input_downcast=True, mode=theano_mode)
        self._test      = theano.function(test_ins, test_loss,
                                     allow_input_downcast=True, mode=theano_mode)
        self._predict  = theano.function(predict_ins, self.y_test,
                                     allow_input_downcast=True, mode=theano_mode, on_unused_input='ignore')

    # train data, adapted from keras.layers.containers.Sequential
    def fit(self, X, scores, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):
        X = standardize_X(X)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
                val_f = self._test

        f = self._train
        out_labels = ['loss']

        ins = X + [scores]
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        return self._fit(f, ins, out_labels=out_labels, batch_size=batch_size, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks,
                         shuffle=False, metrics=metrics)