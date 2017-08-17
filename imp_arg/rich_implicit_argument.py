from copy import deepcopy

import numpy as np

from rich_script.core_argument import CoreArgument
from rich_script.rich_entity import EntitySalience
from rich_tree_pointer import RichTreePointer
from util import Word2VecModel, get_class_name


class RichCandidate(object):
    def __init__(self, arg_pointer, dice_score, core, entity_salience):
        assert isinstance(arg_pointer, RichTreePointer), \
            'arg_pointer must be a {} instance'.format(
                get_class_name(RichTreePointer))
        self.arg_pointer = arg_pointer

        self.dice_score = dice_score

        assert isinstance(core, CoreArgument), \
            'core must be a {} instance'.format(get_class_name(CoreArgument))
        self.core = deepcopy(core)

        assert isinstance(entity_salience, EntitySalience) or \
            entity_salience is None, \
            'entity_salience must be a {} instance or None'.format(
                get_class_name(EntitySalience))
        self.entity_salience = entity_salience

    def get_index(self, model, arg_type='', use_unk=True):
        return self.core.get_index(model, arg_type, use_unk=use_unk)

    @classmethod
    def build(cls, candidate, imp_args, corenlp_reader, use_lemma=True,
              use_entity=True, use_corenlp_tokens=True):

        arg_pointer = candidate.arg_pointer.copy(
            include_treebank=(not use_corenlp_tokens),
            include_corenlp=use_corenlp_tokens)

        dice_score = candidate.dice_score(
            imp_args, use_corenlp_tokens=use_corenlp_tokens)

        core = candidate.arg_pointer.get_core_argument(
            corenlp_reader, use_lemma=use_lemma, use_entity=use_entity)

        entity_salience = candidate.arg_pointer.get_entity_salience(
            corenlp_reader, use_entity=use_entity)

        return cls(arg_pointer, dice_score, core, entity_salience)


class RichImplicitArgument(object):
    def __init__(self, label, arg_type, fillers, rich_candidate_list):
        self.label = label
        self.arg_type = arg_type
        self.fillers = fillers

        if len(self.fillers) > 0:
            self.exist = True
        else:
            self.exist = False

        self.rich_candidate_list = rich_candidate_list
        self.candidate_wv_list = []
        self.valid_candidate_idx_list = []

        self.has_coherence_score = False
        self.coherence_score_list = []
        self.max_coherence_score = 0.0
        self.max_coherence_score_idx = -1

    def get_arg_idx(self):
        if self.arg_type == 'SUBJ':
            return 1
        elif self.arg_type == 'OBJ':
            return 2
        else:
            return 3

    def get_index(self, model, include_type=True, use_unk=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))

        for rich_candidate in self.rich_candidate_list:
            candidate_wv = rich_candidate.get_index(
                model, self.arg_type if include_type else '', use_unk=use_unk)
            self.candidate_wv_list.append(candidate_wv)

        self.valid_candidate_idx_list = \
            [candidate_idx for candidate_idx, candidate_wv
             in enumerate(self.candidate_wv_list) if candidate_wv != -1]

    def set_coherence_score_list(self, coherence_score_list):
        assert len(coherence_score_list) == len(self.rich_candidate_list)
        self.has_coherence_score = True
        self.coherence_score_list = coherence_score_list
        self.max_coherence_score = self.coherence_score_list.max()

        max_indices = \
            [idx for idx, score in enumerate(self.coherence_score_list)
             if score == self.max_coherence_score]

        max_dice_list = np.asarray(
            [self.rich_candidate_list[idx].dice_score
             for idx in max_indices])
        self.max_coherence_score_idx = max_indices[max_dice_list.argmax()]

    def get_max_dice_score(self):
        assert self.max_coherence_score_idx != -1
        return self.rich_candidate_list[self.max_coherence_score_idx].dice_score

    @classmethod
    def build(cls, label, arg_type, fillers, candidates, corenlp_reader,
              use_lemma=True, use_entity=True, use_corenlp_tokens=True):
        rich_candidates = [
            RichCandidate.build(
                candidate, fillers, corenlp_reader,
                use_lemma=use_lemma, use_entity=use_entity,
                use_corenlp_tokens=use_corenlp_tokens)
            for candidate in candidates]

        return cls(label, arg_type, fillers, rich_candidates)
