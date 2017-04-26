import re

import document
from entity import Entity
from token import Token, ParseTokenError
from util import unescape, get_class_name, consts


class Argument(Token):
    # TODO: determine whether or not to include ner in Argument
    def __init__(self, word, lemma, pos, ner='', entity_idx=-1, mention_idx=-1):
        super(Argument, self).__init__(word, lemma, pos)
        self.ner = ner
        if not (isinstance(entity_idx, int) and entity_idx >= -1):
            raise ParseTokenError(
                'entity_idx must be a natural number or -1 (no entity)')
        self.entity_idx = entity_idx
        if not isinstance(mention_idx, int):
            raise ParseTokenError('mention_idx must be an integer')
        if (not (entity_idx == -1 and mention_idx == -1)) \
                and (not (entity_idx >= 0 and mention_idx >= 0)):
            raise ParseTokenError(
                'mention_idx must be a natural number when entity_idx is set, '
                'or -1 when entity_idx is -1')
        self.mention_idx = mention_idx

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos \
               and self.entity_idx == other.entity_idx \
               and self.mention_idx == other.mention_idx

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_entity=False, entity_list=None,
                           use_ner=True, use_lemma=True):
        assert (not use_entity) or entity_list, \
            'entity_list cannot be None or empty when use_entity is specified'
        if use_entity and self.entity_idx != -1:
            assert 0 <= self.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(self.entity_idx)
            assert all(isinstance(entity, Entity) for entity in entity_list), \
                'entity_list must contains only Entity element'
            return entity_list[self.entity_idx].get_representation(
                use_ner, use_lemma)
        # TODO: return self.ner when self.ner is a valid ner tag
        elif use_ner and self.ner != '':
            return self.ner
        else:
            return super(Argument, self).get_representation(use_lemma)

    def get_entity(self, entity_list):
        if self.entity_idx != -1:
            assert self.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(self.entity_idx)
            return entity_list[self.entity_idx]
        return None

    def get_mention(self, entity_list):
        entity = self.get_entity(entity_list)
        if entity:
            assert self.mention_idx < len(entity.mentions), \
                'mention_idx {} out of range'.format(self.mention_idx)
            return entity.get_mention(self.mention_idx)
        return None

    def to_text(self):
        text = super(Argument, self).to_text()
        if self.entity_idx != -1 and self.mention_idx != -1:
            text += '//entity-{}-{}'.format(self.entity_idx, self.mention_idx)
        return text

    arg_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)'
        r'((?://entity-)(?P<entity_idx>\d+)(?:-)(?P<mention_idx>\d+))?$')

    @classmethod
    def from_text(cls, text):
        match = cls.arg_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Argument from {}'.format(text))
        groups = match.groupdict()

        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos']),
            int(groups['entity_idx']) if groups['entity_idx'] else -1,
            int(groups['mention_idx']) if groups['mention_idx'] else -1
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
        ner = token.ner
        if ner not in consts.VALID_NER_TAGS:
            ner = ''
        coref_idx = token.coref.idx if token.coref else -1
        mention_idx = token.mention.mention_idx if token.mention else -1
        return cls(word, lemma, pos, ner, coref_idx, mention_idx)
