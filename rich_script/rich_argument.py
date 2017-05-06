from argument import Argument
from util import Word2VecModel, get_class_name


class RichArgument(object):
    def __init__(self, arg_type, candidate_text_list, entity_idx, mention_idx):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        assert candidate_text_list, 'candidate_text_list cannot be empty'
        assert 0 <= entity_idx < len(candidate_text_list) or \
               (len(candidate_text_list) == 1 and entity_idx == -1), \
            'entity_idx must be between 0 and len(candidate_text_list), ' \
            'or -1 when len(candidate_text_list) = 1'
        assert mention_idx >= 0 or (entity_idx == -1 and mention_idx == -1), \
            'mention_idx must be >= 0, or -1 when entity_idx = -1'
        # type of argument
        self.arg_type = arg_type
        # list of texts for all candidates of the argument
        self.candidate_text_list = candidate_text_list
        # index of the entity the argument points to, -1 if no entity/mention
        self.entity_idx = entity_idx
        # index of the mention the argument points to, -1 if no entity/mention
        self.mention_idx = mention_idx

        # index of target candidate in the candidate list
        self.target_idx = max(self.entity_idx, 0)
        # boolean flag indicating whether the argument has negative candidates
        self.has_neg = (len(self.candidate_text_list) > 1)
        # list of word2vec indices for all candidates of the argument
        self.candidate_wv_list = []

    @classmethod
    def build(cls, arg_type, arg, entity_list, use_entity=True, use_ner=True,
              use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be a {} instance, {} found'.format(
                get_class_name(Argument), type(arg))
        if arg.entity_idx != -1 and use_entity:
            assert 0 <= arg.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            candidate_text_list = [
                entity.get_representation(use_ner=use_ner, use_lemma=use_lemma)
                for entity in entity_list]
            entity_idx = arg.entity_idx
            mention_idx = arg.mention_idx
        else:
            candidate_text_list = [
                arg.get_representation(use_ner=use_ner,use_lemma=use_lemma)]
            entity_idx = -1
            mention_idx = -1
        return cls(arg_type, candidate_text_list, entity_idx, mention_idx)

    @classmethod
    def build_with_vocab_list(cls, arg_type, arg, arg_vocab_list,
                              ner_vocab_list, entity_list, use_entity=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be a {} instance, {} found'.format(
                get_class_name(Argument), type(arg))
        if arg.entity_idx != -1 and use_entity:
            assert 0 <= arg.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            candidate_text_list = [
                entity.get_repr_universal(arg_vocab_list, ner_vocab_list)
                for entity in entity_list]
            entity_idx = arg.entity_idx
            mention_idx = arg.mention_idx
        else:
            candidate_text_list = [
                arg.get_repr_universal(arg_vocab_list, ner_vocab_list)]
            entity_idx = -1
            mention_idx = -1
        return cls(arg_type, candidate_text_list, entity_idx, mention_idx)

    # get word2vec indices for all candidates
    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.candidate_wv_list = []
        for candidate_text in self.candidate_text_list:
            if include_type:
                candidate_wv = model.get_word_index(
                    candidate_text + '-' + self.arg_type)
                # backtrack when prep-pobj is out-of-vocab
                if candidate_wv == -1 and self.arg_type.startswith('PREP'):
                    candidate_wv = model.get_word_index(
                        candidate_text + '-PREP')
            else:
                candidate_wv = model.get_word_index(candidate_text)
            self.candidate_wv_list.append(candidate_wv)
        # remove non-indexed candidates from candidate_text_list and
        # candidate_wv_list, also reset target_idx
        effective_candidate_idx_list = \
            [candidate_idx for candidate_idx, candidate_wv
             in enumerate(self.candidate_wv_list)
             if candidate_wv != -1 or candidate_idx == self.target_idx]
        self.candidate_text_list = \
            [self.candidate_text_list[candidate_idx] for candidate_idx
             in effective_candidate_idx_list]
        self.candidate_wv_list = \
            [self.candidate_wv_list[candidate_idx] for candidate_idx
             in effective_candidate_idx_list]
        self.target_idx = effective_candidate_idx_list.index(self.target_idx)
        if self.candidate_wv_list[self.target_idx] == -1 or \
                        len(self.candidate_wv_list) <= 1:
            self.has_neg = False

    # get the text for the positive candidate
    def get_pos_text(self, include_type=True):
        pos_text = self.candidate_text_list[self.target_idx]
        if include_type:
            pos_text += '-' + self.arg_type
        return pos_text

    # get the list of texts for all negative candidates, might be empty
    def get_neg_text_list(self, include_type=True):
        if len(self.candidate_text_list) == 1:
            return []
        neg_text_list = \
            self.candidate_text_list[self.target_idx] + \
            self.candidate_text_list[self.target_idx+1:]
        if include_type:
            neg_text_list = \
                [neg_text + '-' + self.arg_type for neg_text in neg_text_list]
        return neg_text_list

    # get the word2vec index for the positive candidate, might be -1
    def get_pos_wv(self):
        return self.candidate_wv_list[self.target_idx]

    # get the word2vec indices for all negative candidates, might be empty
    def get_neg_wv_list(self):
        if len(self.candidate_wv_list) == 1:
            return []
        neg_wv_list = \
            self.candidate_wv_list[:self.target_idx] + \
            self.candidate_wv_list[self.target_idx+1:]
        return neg_wv_list

    # boolean flag indicating whether the argument points to an entity
    def has_entity(self):
        return self.entity_idx != -1

    # boolean flag indicating whether the argument points to the first mention
    def is_first_mention(self):
        return self.mention_idx == 0
