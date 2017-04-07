from gensim.models import KeyedVectors


class Word2VecModel(object):
    def __init__(self, name=''):
        self.name = name
        self.word2vec = None
        self.dimension = -1
        self.vocab_size = -1

    def load_model(self, fname, fvocab=None, binary=True):
        self.word2vec = KeyedVectors.load_word2vec_format(
            fname, fvocab=fvocab, binary=binary)
        self.vocab_size, self.dimension = self.word2vec.syn0.shape
        self.word2vec.init_sims()

    def get_word_index(self, word):
        if word == '':
            return -1
        word_vocab = self.word2vec.vocab.get(word)
        if word_vocab is not None:
            return word_vocab.index
        else:
            return -1

    def get_word_vec(self, word, use_norm=True):
        if word == '':
            return None
        try:
            return self.word2vec.word_vec(word, use_norm=use_norm)
        except KeyError:
            return None

    def get_index_vec(self, index, use_norm=True):
        if index < 0 or index >= self.vocab_size:
            return None
        if use_norm:
            return self.word2vec.syn0norm[index]
        else:
            return self.word2vec.syn0[index]
