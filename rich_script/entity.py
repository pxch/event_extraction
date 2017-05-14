from collections import Counter

import document
from token import Token
from util import consts, get_class_name
from core_argument import CoreArgument


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx, head_token_idx,
                 rep, tokens, ner=''):
        self.sent_idx = sent_idx
        if not (0 <= start_token_idx <= head_token_idx < end_token_idx):
            raise ParseEntityError(
                'head_token_idx {} must be between start_token_idx {} '
                'and end_token_idx {}'.format(
                    head_token_idx, start_token_idx, end_token_idx
                ))
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.head_token_idx = head_token_idx
        if not isinstance(rep, bool):
            raise ParseEntityError(
                'rep must be a boolean value, {} found'.format(rep))
        self.rep = rep
        if not len(tokens) == end_token_idx - start_token_idx:
            raise ParseEntityError(
                'number of tokens {} does not match start_token_idx {} '
                'and end_token_idx {}'.format(
                    len(tokens), start_token_idx, end_token_idx))
        if not all(isinstance(token, Token) for token in tokens):
            raise ParseEntityError(
                'every token must be a {} instance, found {}'.format(
                    get_class_name(Token), [type(token) for token in tokens]))
        self.tokens = tokens
        self._head_token = \
            self.tokens[self.head_token_idx - self.start_token_idx]
        if not (ner == '' or ner in consts.VALID_NER_TAGS):
            raise ParseEntityError('{} is not a valid ner tag'.format(ner))
        self.ner = ner

    def __eq__(self, other):
        return self.sent_idx == other.sent_idx and \
               self.start_token_idx == other.start_token_idx and \
               self.end_token_idx == other.end_token_idx and \
               self.head_token_idx == other.head_token_idx and \
               self.rep == other.rep and self.ner == other.ner and \
               all(token == other_token for token, other_token
                   in zip(self.tokens, other.tokens)) and \
               self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def head_token(self):
        return self._head_token

    def get_core_argument(self, use_lemma=True):
        word = self._head_token.get_representation(use_lemma=use_lemma)
        return CoreArgument(word, self.ner)

    def to_text(self):
        return '{}:{}:{}:{}:{}:{}:{}'.format(
            self.sent_idx,
            self.start_token_idx,
            self.end_token_idx,
            self.head_token_idx,
            1 if self.rep else 0,
            self.ner if self.ner != '' else 'NONE',
            ':'.join([token.to_text() for token in self.tokens])
        )

    @classmethod
    def from_text(cls, text):
        parts = [p.strip() for p in text.split(':')]
        if len(parts) < 7:
            raise ParseEntityError(
                'expected at least 7 parts, separated by ;, got {}: {}'.format(
                    len(parts), text))
        sent_idx = int(parts[0])
        start_token_idx = int(parts[1])
        end_token_idx = int(parts[2])
        head_token_idx = int(parts[3])
        rep = True if int(parts[4]) == 1 else False
        ner = parts[5] if parts[5] != 'NONE' else ''
        tokens = [Token.from_text(token_text.strip()) for token_text
                  in parts[6:]]
        return cls(sent_idx, start_token_idx, end_token_idx, head_token_idx,
                   rep, tokens, ner)

    @classmethod
    def from_mention(cls, mention):
        if not isinstance(mention, document.Mention):
            raise ParseEntityError(
                'from_mention must be called with a {} instance'.format(
                    get_class_name(document.Mention)))
        # NOBUG: just use ner of the head token, should be correct
        ner = mention.head_token.ner
        if ner not in consts.VALID_NER_TAGS:
            ner = ''
        return cls(
            mention.sent_idx,
            mention.start_token_idx,
            mention.end_token_idx,
            mention.head_token_idx,
            mention.rep,
            [Token.from_token(token) for token in mention.tokens],
            ner
        )


class Entity(object):
    def __init__(self, mentions):
        if not mentions:
            raise ParseEntityError('must provide at least one mention')
        if not all(isinstance(mention, Mention) for mention in mentions):
            raise ParseEntityError('every mention must be a {} instance'.format(
                get_class_name(Mention)))
        self.mentions = mentions
        self._rep_mention = None
        for mention in self.mentions:
            if mention.rep:
                if self._rep_mention is None:
                    self._rep_mention = mention
                else:
                    raise ParseEntityError(
                        'cannot have more than one representative mentions')
        if self._rep_mention is None:
            raise ParseEntityError('no representative mention provided')
        # NOBUG: set self.ner to be the most frequent ner of all mentions
        # might be different than the ner of rep_mention
        ner_counter = Counter()
        for mention in self.mentions:
            if mention.ner != '':
                ner_counter[mention.ner] += 1
        if len(ner_counter):
            self.ner = ner_counter.most_common(1)[0][0]
        else:
            self.ner = ''

    def __eq__(self, other):
        return all(mention == other_mention for mention, other_mention
                   in zip(self.mentions, other.mentions))

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def rep_mention(self):
        return self._rep_mention

    def get_core_argument(self, use_lemma=True):
        # FIXME: self.ner might be different from _rep_mention.ner
        return self._rep_mention.get_core_argument(use_lemma=use_lemma)

    def to_text(self):
        return ' :: '.join([mention.to_text() for mention in self.mentions])

    @classmethod
    def from_text(cls, text):
        return cls([Mention.from_text(mention_text.strip())
                    for mention_text in text.split(' :: ')])

    @classmethod
    def from_coref(cls, coref):
        if not isinstance(coref, document.Coreference):
            raise ParseEntityError(
                'from_coref must be called with a {} instance'.format(
                    get_class_name(document.Coreference)))
        return cls([Mention.from_mention(mention)
                    for mention in coref.mentions])


class ParseEntityError(Exception):
    pass
