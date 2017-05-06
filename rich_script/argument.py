import re
from warnings import warn

import document
from token import Token, ParseTokenError
from util import unescape, get_class_name, consts


class Argument(Token):
    def __init__(self, word, lemma, pos, ner='', entity_idx=-1, mention_idx=-1):
        super(Argument, self).__init__(word, lemma, pos)
        if not (ner == '' or ner in consts.VALID_NER_TAGS):
            raise ParseTokenError('{} is not a valid ner tag'.format(ner))
        self.ner = ner
        if not (isinstance(entity_idx, int) and entity_idx >= -1):
            raise ParseTokenError(
                'entity_idx must be a natural number or -1 (no entity)')
        self.entity_idx = entity_idx
        if (not isinstance(mention_idx, int)) or \
                ((not (entity_idx == -1 and mention_idx == -1)) and
                 (not (entity_idx >= 0 and mention_idx >= 0))):
            raise ParseTokenError(
                'mention_idx must be a natural number when entity_idx '
                'is not -1, or -1 when entity_idx is -1')
        self.mention_idx = mention_idx

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos and self.ner == other.ner \
               and self.entity_idx == other.entity_idx \
               and self.mention_idx == other.mention_idx

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_ner=True, use_lemma=True):
        if self.entity_idx != -1:
            warn('Calling Argument.get_representation when Argument.entity_idx '
                 'is not -1, should call Entity.get_representation instead')
        if use_ner and self.ner != '':
            return self.ner
        else:
            return super(Argument, self).get_representation(use_lemma=use_lemma)

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
        text += '/{}'.format(self.ner if self.ner != '' else 'NONE')
        if self.entity_idx != -1 and self.mention_idx != -1:
            text += '//entity-{}-{}'.format(self.entity_idx, self.mention_idx)
        return text

    arg_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)/(?P<ner>[^/]*)'
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
            groups['ner'] if groups['ner'] != 'NONE' else '',
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
