from argument import Argument
from util import Word2VecModel, get_class_name
from core_argument import CoreArgument
import abc
from rich_entity import RichEntity


class BaseRichArgument(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, arg_type, core):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        # type of argument
        self.arg_type = arg_type
        assert isinstance(core, CoreArgument), \
            'BaseRichArgument must be initialized with a {} instance'.format(
                get_class_name(CoreArgument))
        # word and ner of the true argument
        self.core = core
        # word2vec index of the true argument
        self.core_wv = -1

    @staticmethod
    def build(arg_type, arg, rich_entity_list, use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be a {} instance, {} found'.format(
                get_class_name(Argument), type(arg))
        if arg.entity_idx == -1:
            core = arg.get_core_argument(use_lemma=use_lemma)
            return RichArgument(arg_type, core)
        else:
            assert 0 <= arg.entity_idx < len(rich_entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            core = rich_entity_list[arg.entity_idx].core
            return RichArgumentWithEntity(
                arg_type, core, rich_entity_list, arg.entity_idx,
                arg.mention_idx)

    # get word2vec indices for all candidates
    @abc.abstractmethod
    def get_index(self, model, include_type=True):
        return

    # get the text for the positive candidate
    @abc.abstractmethod
    def get_pos_text(self, arg_vocab_list=None, ner_vocab_list=None,
                     include_type=True):
        return

    # get the word2vec index for the positive candidate, might be -1
    @abc.abstractmethod
    def get_pos_wv(self):
        return

    # get the list of texts for all negative candidates, might be empty
    @abc.abstractmethod
    def get_neg_text_list(self, arg_vocab_list=None, ner_vocab_list=None,
                          include_type=True):
        return

    # get the word2vec indices for all negative candidates, might be empty
    @abc.abstractmethod
    def get_neg_wv_list(self):
        return

    # get the word2vec indices for all candidates (both positive and negative)
    @abc.abstractmethod
    def get_all_wv_list(self):
        return

    # get the index of positive wv in self.get_all_wv_list()
    @abc.abstractmethod
    def get_target_idx(self):
        return

    # boolean flag indicating whether the argument has negative candidates
    @abc.abstractmethod
    def has_neg(self):
        return

    # boolean flag indicating whether the argument points to the first mention
    @abc.abstractmethod
    def is_first_mention(self):
        return


class RichArgument(BaseRichArgument):
    def __init__(self, arg_type, core):
        super(RichArgument, self).__init__(arg_type, core)

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.core_wv = \
            self.core.get_index(model, self.arg_type if include_type else '')

    def get_pos_text(self, arg_vocab_list=None, ner_vocab_list=None,
                     include_type=True):
        pos_text = self.core.get_text_with_vocab_list(
            arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)
        if include_type:
            pos_text += '-' + self.arg_type
        return pos_text

    def get_pos_wv(self):
        return self.core_wv

    def get_neg_text_list(self, arg_vocab_list=None, ner_vocab_list=None,
                          include_type=True):
        return []

    def get_neg_wv_list(self):
        return []

    def get_all_wv_list(self):
        return [self.core_wv]

    def get_target_idx(self):
        return 0

    def has_neg(self):
        return False

    def is_first_mention(self):
        return False


class RichArgumentWithEntity(BaseRichArgument):
    def __init__(self, arg_type, core, rich_entity_list, entity_idx,
                 mention_idx):
        super(RichArgumentWithEntity, self).__init__(arg_type, core)
        assert rich_entity_list, 'rich_entity_list cannot be empty'
        assert all(isinstance(rich_entity, RichEntity) for rich_entity
                   in rich_entity_list), \
            'all rich_entity must be a {} instance'.format(
                get_class_name(RichEntity))
        # list of all rich entities in the script
        self.rich_entity_list = rich_entity_list
        assert 0 <= entity_idx <= len(rich_entity_list), \
            'entity_idx out of range [0, {})'.format(len(rich_entity_list))
        assert core == rich_entity_list[entity_idx].core, \
            'core = {} not consistent with ' \
            'rich_entity_list[{}].core = {}'.format(
                core, entity_idx, rich_entity_list[entity_idx].core)
        # index of the rich entity that the argument points to
        self.entity_idx = entity_idx
        assert mention_idx >= 0, 'mention_idx must be >= 0'
        # index of the mention that the argument points to
        self.mention_idx = mention_idx
        # word2vec index of all rich entities
        self.entity_wv_list = []
        # list of indices of entities with valid word2vec index (not -1)
        self.valid_entity_idx_list = []

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.core_wv = \
            self.core.get_index(model, self.arg_type if include_type else '')
        for rich_entity in self.rich_entity_list:
            self.entity_wv_list.append(rich_entity.get_index(
                model, self.arg_type if include_type else ''))
        self.valid_entity_idx_list = \
            [entity_idx for entity_idx, entity_wv
             in enumerate(self.entity_wv_list) if entity_wv != -1]

    def get_pos_text(self, arg_vocab_list=None, ner_vocab_list=None,
                     include_type=True):
        pos_text = \
            self.rich_entity_list[self.entity_idx].get_text_with_vocab_list(
                arg_vocab_list=arg_vocab_list, ner_vocab_list=ner_vocab_list)
        if include_type:
            pos_text += '-' + self.arg_type
        return pos_text

    def get_pos_wv(self):
        return self.entity_wv_list[self.entity_idx]

    def get_neg_text_list(self, arg_vocab_list=None, ner_vocab_list=None,
                          include_type=True):
        neg_text_list = []
        for entity_idx in self.valid_entity_idx_list:
            if entity_idx == self.entity_idx:
                continue
            neg_text = \
                self.rich_entity_list[entity_idx].get_text_with_vocab_list(
                    arg_vocab_list=arg_vocab_list,
                    ner_vocab_list=ner_vocab_list)
            if include_type:
                neg_text += '-' + self.arg_type
            neg_text_list.append(neg_text)
        return neg_text_list

    def get_neg_wv_list(self):
        neg_wv_list = []
        for entity_idx in self.valid_entity_idx_list:
            if entity_idx == self.entity_idx:
                continue
            neg_wv_list.append(self.entity_wv_list[entity_idx])
        return neg_wv_list

    def get_all_wv_list(self):
        all_wv_list = []
        for entity_idx in self.valid_entity_idx_list:
            all_wv_list.append(self.entity_wv_list[entity_idx])
        return all_wv_list

    def get_target_idx(self):
        return self.valid_entity_idx_list.index(self.entity_idx)

    def has_neg(self):
        return len(self.valid_entity_idx_list) > 1 and self.get_pos_wv() != -1

    def is_first_mention(self):
        return self.mention_idx == 0
