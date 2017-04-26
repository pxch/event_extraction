import re

import document
from util import escape, unescape, get_class_name


class Token(object):
    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_lemma=True):
        # TODO: return empty string when token is not noun or verb
        if self.pos[:2] not in ['NN', 'VB']:
            return ''
        if use_lemma:
            return self.lemma.lower()
        else:
            return self.word.lower()

    def to_text(self):
        return '{}/{}/{}'.format(
            escape(self.word), escape(self.lemma), escape(self.pos))

    token_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)$')

    @classmethod
    def from_text(cls, text):
        match = cls.token_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Token from {}'.format(text))
        groups = match.groupdict()
        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos'])
        )

    @classmethod
    def from_token(cls, token):
        if not isinstance(token, document.Token):
            raise ParseTokenError(
                'from_token must be called with a {} instance'.format(
                    get_class_name(document.Token)))
        word = token.word
        lemma = token.lemma
        pos = token.pos
        return cls(word, lemma, pos)


class ParseTokenError(Exception):
    pass
