from argument import Argument
from util import Word2VecModel


class RichArgument(object):
    def __init__(self, arg_type, pos_text, neg_text_list):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        self.arg_type = arg_type
        self.pos_text = pos_text
        self.neg_text_list = neg_text_list
        self.has_neg = (len(self.neg_text_list) != 0)
        self.pos_idx = -1
        self.neg_idx_list = []

    @classmethod
    def build(cls, arg_type, arg, entity_list, use_entity=True, use_ner=True,
              use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be an Argument instance, {} found'.format(type(arg))
        if arg.entity_idx != -1 and use_entity:
            assert 0 <= arg.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            entity_text_list = [
                entity.get_representation(use_ner=use_ner, use_lemma=use_lemma)
                for entity in entity_list]
            pos_text = entity_text_list[arg.entity_idx]
            neg_text_list = \
                entity_text_list[:arg.entity_idx] + \
                entity_text_list[arg.entity_idx + 1:]
        else:
            pos_text = arg.get_representation(
                use_entity=False, entity_list=None,
                use_ner=use_ner, use_lemma=use_lemma)
            neg_text_list = []
        return cls(arg_type, pos_text, neg_text_list)

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a Word2VecModel instance'
        if include_type:
            self.pos_idx = model.get_word_index(
                self.pos_text + '-' + self.arg_type)
            self.neg_idx_list = [
                model.get_word_index(neg_text + '-' + self.arg_type)
                for neg_text in self.neg_text_list]
        else:
            self.pos_idx = model.get_word_index(self.pos_text)
            self.neg_idx_list = [
                model.get_word_index(neg_text)
                for neg_text in self.neg_text_list]
        # remove non-indexed negative index
        self.neg_idx_list = [idx for idx in self.neg_idx_list if idx != -1]
        # set self.has_neg to False if self.pos_idx is -1
        # or self.neg_idx_list is empty
        if self.pos_idx == -1 or len(self.neg_idx_list) == 0:
            self.has_neg = False

