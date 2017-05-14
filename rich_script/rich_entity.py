from core_argument import CoreArgument
from entity import Entity
from util import get_class_name


class EntitySalience(object):
    def __init__(self, kwargs):
        self.first_loc = kwargs['first_loc']
        self.head_count = kwargs['head_count']
        self.num_mentions_named = kwargs['num_mentions_named']
        self.num_mentions_nominal = kwargs['num_mentions_nominal']
        self.num_mentions_pronominal = kwargs['num_mentions_pronominal']
        self.num_mentions_total = kwargs['num_mentions_total']

    def get_feature_list(self):
        return [
            self.first_loc,
            self.head_count,
            self.num_mentions_named,
            self.num_mentions_nominal,
            self.num_mentions_pronominal,
            self.num_mentions_total
        ]


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
        kwargs = {
            # get the sentence index of which the first mention is located
            'first_loc': entity.mentions[0].sent_idx,
            # get the count of 3 types of mentions
            'head_count': token_count_dict.get(core.get_word(), 0),
            # initialize number of named mentions to be 0
            'num_mentions_named': 0,
            # initialize number of nominal mentions to be 0
            'num_mentions_nominal': 0,
            # initialize number of pronominal mentions to be 0
            'num_mentions_pronominal': 0,
            # initialize number of total mentions to be 0
            'num_mentions_total': 0
        }
        # count different types of mentions
        for mention in entity.mentions:
            kwargs['num_mentions_total'] += 1
            # add num_mentions_named if mention.ner is not empty
            if mention.ner != '':
                kwargs['num_mentions_named'] += 1
            # add num_mentions_nominal if mention.pos starts with NN
            elif mention.head_token.pos.startswith('NN'):
                kwargs['num_mentions_nominal'] += 1
            # add num_mentions_pronominal if mention.pos starts with PRP
            elif mention.head_token.pos.startswith('PRP'):
                kwargs['num_mentions_pronominal'] += 1
        salience = EntitySalience(kwargs)
        return cls(core, salience)

    def get_index(self, model, arg_type=''):
        return self.core.get_index(model, arg_type)

    def get_text_with_vocab_list(self, arg_vocab_list=None,
                                 ner_vocab_list=None):
        return self.core.get_text_with_vocab_list(
            arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)
