from gensim.models import Word2Vec
import numpy as np

word2vec_path = './GoogleNews-vectors-negative300.bin.gz'


class Embedding:
    def __init__(self):
        self.model = None
        self.dimension = 0

    def load_model(self, path=word2vec_path, zipped=True, ue='strict'):
        print 'Loading word2vec model from: {}, zipped = {}'.format(path, zipped)
        self.model = Word2Vec.load_word2vec_format(path, binary=zipped, unicode_errors=ue)
        self.dimension = self.model.vector_size
        print 'Done\n'

    def zeros(self):
        return np.zeros(self.dimension)

    def get_embedding(self, string):
        try:
            return self.model[string]
        except KeyError:
            return None
