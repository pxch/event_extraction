from collections import Counter

import document
from token import Token
from util import consts, get_class_name


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx, head_token_idx,
                 rep, tokens, ner=''):
        self.sent_idx = sent_idx
        if not (start_token_idx <= head_token_idx < end_token_idx):
            raise ParseMentionError(
                'head_token_idx {} must be between start_token_idx {} '
                'and end_token_idx {}'.format(
                    head_token_idx, start_token_idx, end_token_idx
                ))
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.head_token_idx = head_token_idx
        if not isinstance(rep, bool):
            raise ParseMentionError('rep must be a boolean value')
        self.rep = rep
        if not tokens:
            raise ParseMentionError('must provide at least one token')
        if not all(isinstance(token, Token) for token in tokens):
            raise ParseMentionError(
                'every token must be a {} instance, found {}'.format(
                    get_class_name(Token), [type(token) for token in tokens]))
        self.tokens = tokens
        self.head_token = \
            self.tokens[self.head_token_idx - self.start_token_idx]
        if not (ner == '' or ner in consts.VALID_NER_TAGS):
            raise ParseMentionError('ner {} is not a valid ner tag'.format(ner))
        self.ner = ner

    def __eq__(self, other):
        return self.sent_idx == other.sent_idx \
               and self.start_token_idx == other.start_token_idx \
               and self.end_token_idx == other.end_token_idx \
               and self.head_token_idx == other.head_token_idx \
               and self.rep == other.rep and self.ner == other.ner \
               and all(token == other_token for token, other_token
                       in zip(self.tokens, other.tokens)) \
               and self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_head_token(self):
        return self.head_token

    def get_ner(self):
        return self.ner

    def get_representation(self, use_ner=True, use_lemma=True):
        if use_ner and self.ner != '':
            return self.ner
        else:
            return self.head_token.get_representation(use_lemma)

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
            raise ParseMentionError(
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
            raise ParseMentionError(
                'from_mention must be called with a {} instance'.format(
                    get_class_name(document.Mention)))
        # FIXME: use ner of head token as ner for the mention, might be wrong
        # TODO: (done) use ner of head token as ner of mention, consistent
        ner = mention.head_token.ner
        # TODO: (done) set non valid ner to empty string, consistent
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


class ParseMentionError(Exception):
    pass


class Entity(object):
    def __init__(self, mentions):
        if not mentions:
            raise ParseEntityError('must provide at least one mention')
        if not all(isinstance(mention, Mention) for mention in mentions):
            raise ParseEntityError('every mention must be a {} instance'.format(
                get_class_name(Mention)))
        self.mentions = mentions
        self.rep_mention = None
        for mention in self.mentions:
            if mention.rep:
                if self.rep_mention is None:
                    self.rep_mention = mention
                else:
                    raise ParseEntityError(
                        'cannot have more than one representative mentions')
        if self.rep_mention is None:
            raise ParseEntityError('no representative mention provided')
        # TODO: just use ner of rep mention
        ner_counter = Counter()
        for mention in self.mentions:
            if mention.ner != '':
                ner_counter[mention.ner] += 1
        if len(ner_counter):
            self.ner = ner_counter.most_common(1)[0][0]
        else:
            # TODO: might switch to use ner of rep mention
            self.ner = ''

    def __eq__(self, other):
        return all(mention == other_mention for mention, other_mention
                   in zip(self.mentions, other.mentions))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_rep_mention(self):
        return self.rep_mention

    def get_representation(self, use_ner=True, use_lemma=True):
        # TODO: ignore self.ner, just use rep_mention's representation
        '''
        if use_ner and self.ner != '':
            return self.ner
        '''
        return self.get_rep_mention().get_representation(use_ner, use_lemma)

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

