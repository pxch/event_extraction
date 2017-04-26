from argument import Argument
from util import Word2VecModel, get_class_name


class RichArgument(object):
    def __init__(self, arg_type, text_list, pos_idx, has_entity, mention_idx):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        # type of argument, can be SUBJ, OBJ, or PREP_*
        self.arg_type = arg_type
        self.text_list = text_list
        # TODO: find a better name for this, to avoid ambiguity
        self.pos_idx = pos_idx
        # TODO: use a better name like has_entity or something
        self.has_neg = (len(self.text_list) > 1)
        self.has_entity = has_entity
        self.mention_idx = mention_idx
        self.idx_list = []

    @classmethod
    def build(cls, arg_type, arg, entity_list, use_entity=True, use_ner=True,
              use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be a {} instance, {} found'.format(
                get_class_name(Argument), type(arg))
        if arg.entity_idx != -1 and use_entity:
            assert 0 <= arg.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            text_list = [
                entity.get_representation(use_ner=use_ner, use_lemma=use_lemma)
                for entity in entity_list]
            pos_idx = arg.entity_idx
            has_entity = True
        else:
            text_list = [
                arg.get_representation(use_entity=False, entity_list=None,
                                       use_ner=use_ner, use_lemma=use_lemma)]
            pos_idx = 0
            has_entity = False
        return cls(arg_type, text_list, pos_idx, has_entity, arg.mention_idx)

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        if include_type:
            self.idx_list = [
                model.get_word_index(text + '-' + self.arg_type)
                for text in self.text_list]
        else:
            self.idx_list = [
                model.get_word_index(text) for text in self.text_list]
        # TODO: add logic to deal with cases when idx_list[pos_idx] is -1

    def get_pos_arg_index(self):
        return self.idx_list[self.pos_idx]

    '''
    def __init__(self, arg_type, pos_text, neg_text_list):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        # type of argument, can be SUBJ, OBJ, or PREP_*
        self.arg_type = arg_type
        # text of the true argument, set to Entity.get_representation()
        # if the argument is pointed to an entity,
        # otherwise set to Argument.get_representation()
        self.pos_text = pos_text
        # list of text for other candidates of the argument,
        # set to the list of representations of other entities in the script
        # if the argument is pointed to an entity, otherwise set to empty list
        self.neg_text_list = neg_text_list
        # boolean flag indicating whether the argument has other candidates
        # i.e., the argument is pointed to an entity and
        # the number of entities in the script is greater than 1
        self.has_neg = (len(self.neg_text_list) != 0)
        # index of the argument text, -1 if
        self.pos_idx = -1
        self.neg_idx_list = []

    @classmethod
    def build(cls, arg_type, arg, entity_list, use_entity=True, use_ner=True,
              use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be a {} instance, {} found'.format(
                get_class_name(Argument), type(arg))
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
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
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
    '''
