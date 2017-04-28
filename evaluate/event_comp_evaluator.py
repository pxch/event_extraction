from base_evaluator import BaseEvaluator
from rich_script import RichScript, Script
from rich_script import SingleTrainingInput
from util import get_class_name
import numpy as np
from event_composition import EventCompositionModel


import logging

logging.basicConfig(format='%(levelname) s : : %(message)s', level=logging.INFO)

logger = logging.getLogger('event_comp_evaluator')


def get_most_coherent(model, arg_type, eval_input_list, context_input_list,
                      use_max_score=True):
    coherence_fn = model.coherence_fn

    coherence_score_list = []
    num_context = len(context_input_list)

    if arg_type == 'SUBJ':
        arg_type_input = np.asarray([0] * num_context).astype(np.int32)
    elif arg_type == 'OBJ':
        arg_type_input = np.asarray([1] * num_context).astype(np.int32)
    elif arg_type.startswith('PREP'):
        arg_type_input = np.asarray([2] * num_context).astype(np.int32)
    else:
        raise ValueError(
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type))

    pred_input_a = np.zeros(num_context, dtype=np.int32)
    subj_input_a = np.zeros(num_context, dtype=np.int32)
    obj_input_a = np.zeros(num_context, dtype=np.int32)
    pobj_input_a = np.zeros(num_context, dtype=np.int32)
    for context_idx, context_input in enumerate(context_input_list):
        assert isinstance(context_input, SingleTrainingInput), \
            'context_input must be a {} instance'.format(
                get_class_name(SingleTrainingInput))
        pred_input_a[context_idx] = context_input.pred_input
        subj_input_a[context_idx] = context_input.subj_input
        obj_input_a[context_idx] = context_input.obj_input
        pobj_input_a[context_idx] = context_input.pobj_input

    for eval_input in eval_input_list:
        assert isinstance(eval_input, SingleTrainingInput), \
            'eval_input must be a {} instance'.format(
                get_class_name(SingleTrainingInput))
        pred_input_b = np.asarray([eval_input.pred_input] * num_context).astype(np.int32)
        subj_input_b = np.asarray([eval_input.subj_input] * num_context).astype(np.int32)
        obj_input_b = np.asarray([eval_input.obj_input] * num_context).astype(np.int32)
        pobj_input_b = np.asarray([eval_input.pobj_input] * num_context).astype(np.int32)

        coherence_output = coherence_fn(
            pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
            pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
            arg_type_input)
        if use_max_score:
            coherence_score_list.append(coherence_output.max())
        else:
            coherence_score_list.append(coherence_output.sum())

    most_coherent_idx = coherence_score_list.index(max(coherence_score_list))
    return most_coherent_idx


class EventCompositionEvaluator(BaseEvaluator):
    def __init__(self, model=None, use_lemma=True, include_neg=True,
                 include_prt=True, use_entity=True, use_ner=True,
                 include_prep=True, include_type=True, use_max_score=True):
        BaseEvaluator.__init__(self)
        assert model is None or isinstance(model, EventCompositionModel), \
            'model must be None or a {} instance'.format(
                get_class_name(EventCompositionModel))
        self.model = model
        if self.model is not None:
            self.word2vec_model = self.model.arg_comp_model.word2vec
        else:
            self.word2vec_model = None
        self.use_lemma = use_lemma
        self.include_neg = include_neg
        self.include_prt = include_prt
        self.use_entity = use_entity
        self.use_ner = use_ner
        self.include_prep = include_prep
        self.include_type = include_type
        self.use_max_score = use_max_score

    def set_model(self, model):
        assert isinstance(model, EventCompositionModel), \
            'model must be a {} instance'.format(
                get_class_name(EventCompositionModel))
        self.model = model
        self.word2vec_model = self.model.arg_comp_model.word2vec

    def print_debug_message(self):
        return 'Evaluation based on word2vec to find most similar event, ' \
               'model = {}, use_lemma = {}, include_neg = {}, ' \
               'include_prt = {}, use_entity = {}, use_ner = {}, ' \
               'include_prep = {}, include_type = {}, ' \
               'ignore_first_mention = {}, use_max_score = {}'.format(
                'event_comp', self.use_lemma, self.include_neg,
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
        rich_script.get_index(self.word2vec_model, self.include_type)
        # remove events with None embedding (pred_idx == -1)
        rich_event_list = [rich_event for rich_event in rich_script.rich_events
                           if rich_event.pred_idx != -1]
        pos_input_list = \
            [rich_event.get_pos_training_input(include_all_pobj=False)
             for rich_event in rich_event_list]
        for event_idx, rich_event in enumerate(rich_event_list):
            logger.debug('Processing event #{}'.format(event_idx))
            context_input_list = \
                pos_input_list[:event_idx] + pos_input_list[event_idx+1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=False)
            for rich_arg, eval_input_list in eval_input_list_all:
                if (not self.ignore_first_mention) or \
                        (not rich_arg.is_first_mention()):
                    most_coherent_idx = get_most_coherent(
                        self.model, rich_arg.arg_type, eval_input_list,
                        context_input_list, self.use_max_score)
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
