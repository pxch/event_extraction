from core_argument import CoreArgument
from entity import Entity
from util import consts, get_class_name


class EntitySalience(object):
    def __init__(self, **kwargs):
        for feature in consts.SALIENCE_FEATURES:
            if feature in kwargs:
                self.__dict__[feature] = kwargs[feature]
            else:
                self.__dict__[feature] = 0

    def get_feature(self, feature):
        assert feature in consts.SALIENCE_FEATURES
        return self.__dict__[feature]

    def get_feature_list(self, feature_list=None):
        if feature_list is None:
            feature_list = consts.SALIENCE_FEATURES
        result = []
        for feature in feature_list:
            result.append(self.get_feature(feature))
        return result

    def to_text(self):
        return ','.join(map(str, self.get_feature_list()))

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) == consts.NUM_SALIENCE_FEATURES, \
            'expecting {} parts separated by ",", found {}'.format(
                consts.NUM_SALIENCE_FEATURES, len(parts))
        kwargs = {}
        for feature, value in zip(consts.SALIENCE_FEATURES, parts):
            kwargs[feature] = int(value)
        return cls(**kwargs)


class RichEntity(object):
    def __init__(self, core, salience):
        assert isinstance(core, CoreArgument), \
            'RichEntity must be initialized with a {} instance'.format(
                get_class_name(CoreArgument))
        self.core = core
        assert isinstance(salience, EntitySalience), \
            'RichEntity must be initialized with a {} instance'.format(
                get_class_name(EntitySalience))
        self.salience = salience

    @classmethod
    def build(cls, entity, token_count_dict, use_lemma=True):
        assert entity is not None and isinstance(entity, Entity), \
            'entity must be a {} instance, {} found'.format(
                get_class_name(Entity), type(entity))
        # get the representation and ner of the entity
        core = entity.get_core_argument(use_lemma=use_lemma)
        # get the sentence index of which the first mention is located
        first_loc = entity.mentions[0].sent_idx
        # get the count of the head word in the document
        head_count = token_count_dict.get(core.get_word(), 0)
        # initialize number of named mentions / nominal mentions /
        # pronominal mentions / total mentions to be 0
        num_mentions_named = 0
        num_mentions_nominal = 0
        num_mentions_pronominal = 0
        num_mentions_total = 0
        # count different types of mentions
        for mention in entity.mentions:
            num_mentions_total += 1
            # add num_mentions_named if mention.ner is not empty
            if mention.ner != '':
                num_mentions_named += 1
            # add num_mentions_nominal if mention.pos starts with NN
            elif mention.head_token.pos.startswith('NN'):
                num_mentions_nominal += 1
            # add num_mentions_pronominal if mention.pos starts with PRP
            elif mention.head_token.pos.startswith('PRP'):
                num_mentions_pronominal += 1
        salience = EntitySalience(
            first_loc=first_loc,
            head_count=head_count,
            num_mentions_named=num_mentions_named,
            num_mentions_nominal=num_mentions_nominal,
            num_mentions_pronominal=num_mentions_pronominal,
            num_mentions_total=num_mentions_total
        )
        return cls(core, salience)

    def get_index(self, model, arg_type='', use_unk=True):
        return self.core.get_index(model, arg_type, use_unk=use_unk)

    def get_text_with_vocab_list(self, arg_vocab_list=None,
                                 ner_vocab_list=None):
        return self.core.get_text_with_vocab_list(
            arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)

    def get_salience(self):
        return self.salience
