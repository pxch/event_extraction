from copy import deepcopy

from base_evaluator import BaseEvaluator
from rich_script import RichArgument, RichScript, Script
from rich_script import SingleTrainingInput, SingleTrainingInputMultiPobj
from util import Word2VecModel, get_class_name, cos_sim
import numpy as np

import logging

logging.basicConfig(format='%(levelname) s : : %(message)s', level=logging.INFO)

logger = logging.getLogger('word2vec_evaluator')


def get_arg_embedding(model, rich_arg):
    assert isinstance(model, Word2VecModel), \
        'model must be a {} instance'.format(get_class_name(Word2VecModel))
    assert isinstance(rich_arg, RichArgument), \
        'rich_arg must be a {} instance'.format(get_class_name(RichArgument))
    pos_embedding = model.get_index_vec(rich_arg.pos_idx)
    neg_embedding_list = \
        [model.get_index_vec(neg_idx) for neg_idx in rich_arg.neg_idx_list]
    return pos_embedding, neg_embedding_list


def get_event_vector(model, event_input, use_all_pobj=True):
    assert isinstance(model, Word2VecModel), \
        'model must be a {} instance'.format(get_class_name(Word2VecModel))
    if use_all_pobj:
        assert isinstance(event_input, SingleTrainingInputMultiPobj), \
            'event_input must be a {} instance when use_all_pobj=True'.format(
                get_class_name(SingleTrainingInputMultiPobj))
    else:
        assert isinstance(event_input, SingleTrainingInput), \
            'event_input must be a {} instance when use_all_pobj=False'.format(
                get_class_name(SingleTrainingInput))
    vector = np.zeros(model.vector_size)
    pred_vector = model.get_index_vec(event_input.pred_input)
    if pred_vector is not None:
        vector += pred_vector
    else:
        return None
    subj_vector = model.get_index_vec(event_input.subj_input)
    if subj_vector is not None:
        vector += subj_vector
    obj_vector = model.get_index_vec(event_input.obj_input)
    if obj_vector is not None:
        vector += obj_vector
    if use_all_pobj:
        for pobj_input in event_input.pobj_input_list:
            pobj_vector = model.get_index_vec(pobj_input)
            if pobj_vector is not None:
                vector += pobj_vector
    else:
        pobj_vector = model.get_index_vec(event_input.pobj_input)
        if pobj_vector is not None:
            vector += pobj_vector
    return vector


def get_coherence_scores(target_vector, context_vector_list):
    return [cos_sim(target_vector, context_vector)
            for context_vector in context_vector_list]


def get_most_coherent(eval_vector_list, context_vector_list,
                      use_max_score=True):
    coherence_score_list = []
    for eval_vector in eval_vector_list:
        if use_max_score:
            coherence_score_list.append(max(
                get_coherence_scores(eval_vector, context_vector_list)))
        else:
            coherence_score_list.append(sum(
                get_coherence_scores(eval_vector, context_vector_list)))

    most_coherent_idx = coherence_score_list.index(max(coherence_score_list))
    return most_coherent_idx


class Word2VecEvaluator(BaseEvaluator):
    def __init__(self, model=None, use_lemma=True, include_neg=True,
                 include_prt=True, use_entity=True, use_ner=True,
                 include_prep=True, include_type=True,
                 ignore_first_mention=False, use_max_score=True):
        BaseEvaluator.__init__(self)
        assert model is None or isinstance(model, Word2VecModel), \
            'word2vec must be None or a {} instance'.format(
                get_class_name(Word2VecModel))
        self.model = model
        self.use_lemma = use_lemma
        self.include_neg = include_neg
        self.include_prt = include_prt
        self.use_entity = use_entity
        self.use_ner = use_ner
        self.include_prep = include_prep
        self.include_type = include_type
        self.ignore_first_mention = ignore_first_mention
        self.use_max_score = use_max_score

    def set_model(self, model):
        assert isinstance(model, Word2VecModel), \
            'word2vec must be a {} instance'.format(
                get_class_name(Word2VecModel))
        self.model = model

    def print_debug_message(self):
        return 'Evaluation based on word2vec to find most similar event, ' \
               'model = {}, use_lemma = {}, include_neg = {}, ' \
               'include_prt = {}, use_entity = {}, use_ner = {}, ' \
               'include_prep = {}, include_type = {}, ' \
               'ignore_first_mention = {}, use_max_score = {}'.format(
                self.model.name, self.use_lemma, self.include_neg,
                self.include_prt, self.use_entity, self.use_ner,
                self.include_prep, self.include_type,
                self.ignore_first_mention, self.use_max_score)

    def evaluate_script(self, script):
        assert isinstance(script, Script), \
            'evaluate_script must be called with a {} instance'.format(
                get_class_name(Script))
        logger.debug('Processing script #{}'.format(script.doc_name))

        rich_script = RichScript.build(
            script,
            use_lemma=self.use_lemma,
            include_neg=self.include_neg,
            include_prt=self.include_prt,
            use_entity=self.use_entity,
            use_ner=self.use_ner,
            include_prep=self.include_prep
        )
        rich_script.get_index(self.model, self.include_type)
        # remove events with None embedding (pred_idx == -1)
        rich_event_list = [rich_event for rich_event in rich_script.rich_events
                           if rich_event.pred_idx != -1]
        pos_input_list = [rich_event.get_pos_training_input_multi_pobj()
                          for rich_event in rich_event_list]
        pos_vector_list = [get_event_vector(self.model, pos_input)
                           for pos_input in pos_input_list]
        for event_idx, rich_event in enumerate(rich_event_list):
            logger.debug('Processing event #{}'.format(event_idx))
            context_vector_list = \
                pos_vector_list[:event_idx] + pos_vector_list[event_idx+1:]

            eval_input_list_all = rich_event.get_eval_input_list_all()
            for rich_arg, eval_input_list in eval_input_list_all:
                eval_vector_list = [get_event_vector(self.model, eval_input)
                                    for eval_input in eval_input_list]
                most_coherent_idx = get_most_coherent(
                    eval_vector_list,
                    context_vector_list,
                    self.use_max_score
                )
                correct = (most_coherent_idx == rich_arg.target_idx)
                num_choices = len(eval_input_list)
                self.eval_stats.add_eval_result(
                    rich_arg.arg_type,
                    correct,
                    num_choices
                )
                logger.debug(
                    'Processing {}, correct = {}, num_choices = {}'.format(
                        rich_arg.arg_type, correct, num_choices))
