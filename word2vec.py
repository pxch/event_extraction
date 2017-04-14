from gensim.models import KeyedVectors
from os.path import basename


class Word2VecModel(object):
    def __init__(self, name, word2vec, dimension, vocab_size):
        self.name = name
        self.word2vec = word2vec
        self.dimension = dimension
        self.vocab_size = vocab_size

    @classmethod
    def load_model(cls, fname, fvocab=None, binary=True):
        name = basename(fname)
        word2vec = KeyedVectors.load_word2vec_format(
            fname, fvocab=fvocab, binary=binary)
        word2vec.init_sims()
        vocab_size, dimension = word2vec.syn0.shape
        return cls(name=name, word2vec=word2vec, dimension=dimension,
                   vocab_size=vocab_size)

    def get_vocab(self):
        return self.word2vec.vocab

    def get_id2word(self):
        return [word for (id, word) in sorted(
            [(v.index, word) for (word, v) in self.word2vec.vocab.items()])]

    @staticmethod
    def build_id2word(vocab):
        return [word for (id, word) in sorted(
            [(v.index, word) for (word, v) in vocab.items()])]

    def get_vector_matrix(self, use_norm=True):
        if use_norm:
            return self.word2vec.syn0norm
        else:
            return self.word2vec.syn0

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
