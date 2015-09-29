import itertools
from keras.preprocessing.sequence import pad_sequences

class Batch(object):

    def __init__(self, vocab, infiles, batch_size):
        self.vocab = vocab
        self.infiles = infiles
        self.batch_size = batch_size
        self.total_batches = 0
        self.current_batch = 0

    def get_data(self, maxlen=30):
        for infile in self.infiles:
            with open(infile, 'r') as f:
                while True:
                    chunk = list(itertools.islice(f, self.batch_size))
                    X = []
                    y = []
                    for line in chunk:
                        sequences, next_words = self.process_line(line, maxlen)
                        X.append(sequences)
                        y.append(next_words)
                    #batch = self.pad_1h_sequences(batch, maxlen=maxlen, padding='post', truncating='post')
                    yield X, y
                    #yield chunk

    def process_line(self, line, maxlen, step=3):
        sequences = []
        next_words = []
        line = line.split(' ')
        for i in range(0, len(line) - maxlen, step):
            sequences.append(line[i : i + maxlen])
            next_words.append(line[i + maxlen])

        return sequences, next_words

        '''
        batch_indices = xrange(0, len(sentences), self.batch_size)
        self.total_batches = len(batch_indices)

        for batch_number, batch_index in enumerate(batch_indices):
            self.current_batch = batch_number
            Xs = []
            Ys = []

            for sentence in sentences[batch_index:batch_index + self.batch_size]:
                Xs.append(self.vocab.sentence_to_1h(sentence))
                Ys.append(self.vocab.sentence_to_ids(sentence))

            Xs = self.pad_1h_sequences(Xs, maxlen=maxlen, padding='post', truncating='post')
            Ys = pad_sequences(Ys, maxlen=maxlen, padding='post', truncating='post')
            yield (Xs, Ys)
        '''

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