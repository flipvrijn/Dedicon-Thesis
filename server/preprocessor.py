from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from sklearn.pipeline import make_pipeline
from gensim.models import Word2Vec

from itertools import islice
import numpy as np

class Preprocessor:
    def __init__(self):
        self._stemmer = SnowballStemmer('english')
        self._tfidf = None
        self._svd = None
        self._w2v = None
        self._raw = None

    def set_tfidf(self, m_tfidf, m_svd=None):
        self._tfidf = m_tfidf
        self._svd = m_svd

    def set_w2v(self, m_w2v):
        self._w2v = m_w2v
        self._stemmer = SnowballStemmer('english')
        # Create stem-to-word dict
        self._stem2word = {}
        for word in self._w2v.index2word:
            stem = self._stemmer.stem(word)
            self._stem2word[stem] = word

    def set_raw(self, m_raw):
        self._raw = m_raw

    def prepare_text(self, text, stem=True, dictionary=None):
        output = wordpunct_tokenize(text)
        if stem:
            if dictionary:
                output = [self._stemmer.stem(word) for word in output if word in dictionary]
            else:
                output = map(self._stemmer.stem, output)
        return output

    def window(self, seq, n):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) < n:
            yield result
        if len(result) == n:
            yield result    
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def do_raw(self, text, with_stemming=False):
        assert self._raw is not None, "Raw model is not set"

        # Transform into count vector
        words = self.prepare_text(text, stem=with_stemming)
        doc = [' '.join(words)]
        vectorized = self._raw.transform(doc)

        # Limit vector to 512 dimensionality
        output = np.zeros((1, 512), np.int64)
        vec = np.array(vectorized[:512])
        output[0, :vec.shape[0]] = vec

        return output

    def do_tfidf(self, text, with_svd=False, with_stemming=True):
        ''' Transforms text to a TF-IDF feature vector 
            Output size: 1 x 512 (with svd) or Y x 512 (without svd)
        '''
        assert self._tfidf is not None, "TF-IDF is not set"

        words = self.prepare_text(text, stem=with_stemming)
        doc = [' '.join(words)]
        output = self._tfidf.transform(doc)
        output = output.todense()
        if with_svd:
            assert self._svd is not None, "SVD is not set"
            output = self._svd.transform(output)
        else:
            # Pad with zeros
            pad_size = 512 - output.shape[1] % 512
            output = np.pad(output, [(0, 0), (0, pad_size)], 'constant')
            # and reshape
            output = output.reshape((output.shape[1] / 512, 512))
        return output

    def do_w2v(self, text, n=2, with_stemming=False):
        ''' Transforms text to a Word2Vec feature vector
            Output size: Y x 512
        '''
        assert self._w2v is not None, "Word2Vec is not set"

        # Create n-grams
        X = []
        for sent in sent_tokenize(text):
            words = self.prepare_text(sent, stem=with_stemming, dictionary=self._w2v)
            if words:
                X += [x for x in self.window(words, n)]

        # Transform n-grams to W2V
        output = None
        for i, ngram in enumerate(X):
            vec = np.array([self._w2v[self._stem2word[w]] for w in ngram]).prod(axis=0)
            if i == 0:
                output = vec 
            else:
                output = np.vstack((output, vec))

        # Clip to max 150
        if output is not None:
            if output.ndim != 2:
                output = output[np.newaxis, :]
            output = output[:150] # magic number :(
            output[:output.shape[0], :output.shape[1]] = output

        return output

    def do_w2vtfidf(self, text, n_best=150):
        assert self._tfidf is not None, "TF-IDF is not set"
        assert self._w2v is not None, "Word2Vec is not set"

        tfidf_feature_names = self._tfidf.get_feature_names()

        # Get TF-IDF count for current context
        tfidf_features = self.do_tfidf(text).tolist()[0]
        # Sort on score and grab best n_best
        scores = sorted(
            [(tfidf_feature_names[pair[0]], pair[1]) for pair in zip(range(0, len(tfidf_features)), tfidf_features) if pair[1] > 0], 
            key=lambda x: x[1] * -1
        )[:n_best]

        output = np.zeros((n_best, 512), dtype=np.float32)
        num_features = 0
        for stem, score in scores:
            # skip n-grams or unknown words
            if ' ' in stem or stem not in self._stem2word:
                continue
            output[num_features] = self._w2v[self._stem2word[stem]]
            num_features += 1

        return output