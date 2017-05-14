import re
from warnings import warn

import document
from token import Token, ParseTokenError
from util import escape, unescape, get_class_name


class Predicate(Token):
    def __init__(self, word, lemma, pos, neg=False, prt=''):
        super(Predicate, self).__init__(word, lemma, pos)
        if not isinstance(neg, bool):
            raise ParseTokenError(
                'neg must be a boolean value in initializing a Predicate')
        self.neg = neg
        self.prt = prt

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos and self.neg == other.neg \
               and self.prt == other.prt

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_full_representation(self, use_lemma=True):
        result = super(Predicate, self).get_representation(use_lemma=use_lemma)
        if self.prt:
            result += '_' + self.prt
        if self.neg:
            result = 'not_' + result
        return result

    def to_text(self):
        text = super(Predicate, self).to_text()
        if self.neg:
            text = 'not//' + text
        if self.prt != '':
            text += '//' + escape(self.prt)
        return text

    pred_re = re.compile(
        r'^((?P<neg>not)(?://))?(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)'
        r'((?://)(?P<prt>[^/]*))?$')

    @classmethod
    def from_text(cls, text):
        match = cls.pred_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Predicate from {}'.format(text))
        groups = match.groupdict()
        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos']),
            True if groups['neg'] is not None else False,
            unescape(groups['prt']) if groups['prt'] is not None else ''
        )

    @classmethod
    def from_token(cls, token, **kwargs):
        if not isinstance(token, document.Token):
            raise ParseTokenError(
                'from_token must be called with a {} instance'.format(
                    get_class_name(document.Token)))
        if not token.pos.startswith('VB'):
            raise ParseTokenError(
                'from_token cannot be called with a {} token'.format(token.pos))
        if 'neg' not in kwargs or (not isinstance(kwargs['neg'], bool)):
            raise ParseTokenError(
                'a boolean value neg must be passed in as a keyword parameter')
        word = token.word
        lemma = token.lemma
        pos = token.pos
        prt = ''
        if token.compounds:
            if len(token.compounds) > 1:
                warn('Predicate token should contain at most one compound '
                     '(particle), found {}'.format(len(token.compounds)))
            prt = token.compounds[0].lemma
        return cls(word, lemma, pos, kwargs['neg'], prt)

