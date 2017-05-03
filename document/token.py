from warnings import warn

from util.consts import VALID_NER_TAGS


class Token(object):
    def __init__(self, word, lemma, pos):
        # word form of the token
        self.word = word.encode('ascii', 'ignore')
        # lemma form of the token
        self.lemma = lemma.encode('ascii', 'ignore')
        # part-of-speech tag of the token
        self.pos = pos
        # name entity tag of the token
        self.ner = ''
        # compounds
        self.compounds = []
        # boolean indicator to prevent adding duplicated compound words
        self.compounds_set = False
        # index of the sentence where the token is located
        # set in Sentence.add_token() method
        self.sent_idx = -1
        # index of the token in the sentence
        # set in Sentence.add_token() method
        self.token_idx = -1
        # pointer to the coreference chain which the token belongs to
        self.coref = None
        # pointer to the mention where the token belongs to
        self.mention = None

    def __str__(self):
        return '{}-{}/{}/{}'.format(
            self.token_idx, self.word, self.lemma, self.pos)

    def print_with_coref(self, include_compounds=False):
        result = '{}-{}'.format(self.token_idx, self.word)
        if include_compounds and self.compounds:
            result = '{}_{}'.format(
                result, '_'.join([token.word for token in self.compounds]))
        if self.coref is not None:
            result += '-entity#{:0>3d}'.format(self.coref.idx)
        return result

    # returns True if the token is a common noun or proper noun
    def is_noun(self):
        return self.pos.startswith('NN')

    # returns True if the token is a verb
    def is_verb(self):
        return self.pos.startswith('VB')

    # string form of the token
    def string_form(
            self, use_ner=False, use_lemma=False, include_compounds=False):
        result = ''
        # if use_ner is set, and the token has a valid ner tag, return the tag
        if use_ner and self.ner in VALID_NER_TAGS:
            result = self.ner
        # if the token is a noun or a verb
        elif self.is_noun() or self.is_verb():
            # return the lower case of self.lemma if use_lemma is set
            if use_lemma:
                result = self.lemma.lower()
            # otherwise return the lower case of self.word
            else:
                result = self.word.lower()
            # if include_compounds is set, and the token has compound tokens
            # (currently only particles for verbs), concatenate the word/lemma
            # forms of all compound tokens to the output accordingly
            if include_compounds and self.compounds:
                for compound_token in self.compounds:
                    if use_lemma:
                        result += '_' + compound_token.lemma.lower()
                    else:
                        result += '_' + compound_token.word.lower()
        return result

    def set_attrib(self, attrib, value):
        assert attrib in self.__dict__, \
            '{} is not an attribute of class {}'.format(
                attrib, self.__class__.__name__)
        assert attrib not in ['word', 'lemma', 'pos'], \
            'Resetting the word/lemma/pos of a token after initialization ' \
            + 'is not allowed'
        self.__dict__[attrib] = value

    def add_coref_info(self, coref, mention):
        if self.mention is not None:
            warn('Token {} has existing mention {}'.format(self, self.mention))
            if self.mention.end_token_idx - self.mention.start_token_idx > \
                    mention.end_token_idx - mention.start_token_idx:
                warn(
                    'The new mention {} is nested in the exising mention, '
                    'ignore the new mention'.format(mention))
                return
            else:
                warn('Thew new mention {} has longer span than the existing '
                     'mention, override the existing mention'.format(mention))
        self.coref = coref
        self.mention = mention
