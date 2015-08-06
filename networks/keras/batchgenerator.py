from vocab import Vocab
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from IPython import embed

class BatchGenerator(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def get_data(self, mode='index', batch_size=128, maxlen=30):
        valid_modes = ['original', 'index', 'binary']
        if mode not in valid_modes:
            raise ValueError('Unknown mode: {}. Valid modes are: {}').format(mode, ' '.join(valid_modes))   

        for batch_index in xrange(0, len(self.vocab.sentences), batch_size):
            mat = None
            if mode == 'original':
                mat = []
                for sentence in self.vocab.sentences[batch_index:batch_index + batch_size]:
                    mat.append(sentence)
            if mode == 'index':
                mat = []
                for sentence in self.vocab.sentences[batch_index:batch_index + batch_size]:
                    mat.append(self.vocab.sentence_to_ids(sentence))
                mat = pad_sequences(mat, maxlen=maxlen, padding='post', truncating='post')
            elif mode == 'binary':
                mat = []
                for sentence in self.vocab.sentences[batch_index:batch_index + batch_size]:
                    mat.append(self.vocab.sentence_to_1h(sentence))
                mat = self.pad_1h_sequences(mat, maxlen=maxlen, padding='post', truncating='post')
            yield mat

    def pad_1h_sequences(self, sequences, maxlen=None, dtype='bool', padding='pre', truncating='pre', value=0.):
        """
            Pad each sequence to the same length: 
            the length of the longuest sequence.

            If maxlen is provided, any sequence longer
            than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
            the end of the sequence.

            Supports post-padding and pre-padding (default).

        """
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen, self.vocab.token_count)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x