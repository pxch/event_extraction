import gensim
import numpy as np

word2vec_path = './GoogleNews-vectors-negative300.bin.gz'


class Embedding:
    def __init__(self, name, dimension, syntax_label='',
                 use_ner=False, use_lemma=False, include_compounds=False):
        self.model = None
        self.name = name
        self.dimension = dimension
        self.syntax_label = syntax_label
        self.use_ner = use_ner
        self.use_lemma = use_lemma
        self.include_compounds = include_compounds

    def load_model(self, path=word2vec_path, zipped=True, ue='strict'):
        print 'Loading word2vec model from: {}, zipped = {}'.format(
            path, zipped)
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            path, binary=zipped, unicode_errors=ue)
        # self.dimension = self.model.vector_size
        print 'Done\n'

    def zeros(self):
        return np.zeros(self.dimension)

    def get_embedding(self, string):
        try:
            return self.model[string]
        except KeyError:
            # returns None if out of vocabulary
            return None

    # TODO: add variant of whether or not to use ner and include compounds
    def get_token_embedding(self, token, suffix=''):
        if self.syntax_label:
            assert suffix != '', \
                'Words in the embedding model have syntactic labels, ' \
                'must provide suffix to the token'
        if token.is_noun() or token.is_verb():
            token_string_form = token.string_form(
                    self.use_ner, self.use_lemma, self.include_compounds)
            try:
                return self.get_embedding(token_string_form + suffix)
            except KeyError:
                if suffix.startswith('_PREP'):
                    try:
                        return self.get_embedding(token_string_form + '_PREP')
                    except KeyError:
                        pass
                pass
        '''
        token_word = token.string_form(use_ner=use_ner, use_lemma=False,
                                       include_compounds=include_compounds)
        token_lemma = token.string_form(use_ner=use_ner, use_lemma=True,
                                        include_compounds=include_compounds)
        if token.is_noun() or token.is_verb():
            try:
                return self.get_embedding(token_word + suffix)
            except KeyError:
                try:
                    return self.get_embedding(token_lemma + suffix)
                except KeyError:
                    pass
        '''
        # returns None if out of vocabulary
        return None

    def get_mention_embedding(
            self, mention, suffix='', head_only=False):
        if self.syntax_label:
            assert suffix != '', \
                'Words in the embedding model have syntactic labels, ' \
                'must provide suffix to the mention'
        embedding = self.zeros()
        if head_only:
            head_embedding = self.get_token_embedding(
                mention.head_token, suffix=suffix)
            if head_embedding is not None:
                embedding += head_embedding
        else:
            for token in mention.tokens:
                token_embedding = self.get_token_embedding(token, suffix=suffix)
                if token_embedding is not None:
                    embedding += token_embedding
        return embedding

    def get_coref_embedding(
            self, coref, suffix='', head_only=False, rep_only=True,
            exclude_mention_idx=-1):
        if self.syntax_label:
            assert suffix != '', \
                'Words in the embedding model have syntactic labels, ' \
                'must provide suffix to the coreference'
        embedding = self.zeros()
        if rep_only:
            embedding += self.get_mention_embedding(
                coref.rep_mention, suffix, head_only)
        else:
            for mention_idx, mention in enumerate(coref.mentions):
                if mention_idx != exclude_mention_idx:
                    embedding += self.get_mention_embedding(
                        mention, suffix, head_only)
        return embedding
