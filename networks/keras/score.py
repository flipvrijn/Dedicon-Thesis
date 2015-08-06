import theano.tensor as T
import theano

from IPython import embed

class Score(object):
    def __init__(self, layers):
        ''' Calculate the image/sentence score of the two layers.
        '''
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
        self.layers = layers
        self.params = []
        self.regularizers = []
        self.constraints = []
        for l in self.layers:
            params, regs, consts = l.get_params()
            self.regularizers += regs
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

    def get_params(self):
        return self.params, self.regularizers, self.constraints

    def get_output(self, train=False):
        if len(self.layers) != 2:
            raise Exception('Score can only be calculated with two layers')

        # Shuffling and reshaping inputs to easily compile the
        # max scores for each word in the sentence
        words   = self.layers[1].get_output(train).dimshuffle(0, 2, 1) # (t, nb_samples, embed_dim) -> (t, embed_dim, nb_samples)
        regions = T.shape_padleft(self.layers[0].get_output(train), 1) \
            .repeat(T.shape(words)[0], 0)                            # (nb_samples, embed_dim) -> (t, nb_samples, embed_dim)

        return T.max_and_argmax(T.batched_dot(regions, words), axis=1)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "layers": [l.get_config() for l in self.layers]}