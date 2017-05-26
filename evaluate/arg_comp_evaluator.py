import numpy as np

from base_evaluator import BaseEvaluator
from event_comp_model import EventCompositionModel
from rich_script import IndexedEvent
from util import get_class_name, cos_sim


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

    most_coherent_idx = coherence_score_list.index(
        max(coherence_score_list))
    return most_coherent_idx


class ArgumentCompositionEvaluator(BaseEvaluator):
    def __init__(self, logger=None, use_lemma=True, include_type=True,
                 ignore_first_mention=False, filter_stop_events=True,
                 use_max_score=True):
        super(ArgumentCompositionEvaluator, self).__init__(
            logger=logger,
            use_lemma=use_lemma,
            include_type=include_type,
            include_all_pobj=False,
            ignore_first_mention=ignore_first_mention,
            filter_stop_events=filter_stop_events
        )
        self.use_max_score = use_max_score
        self.model_name = 'argument_composition'

    def set_model(self, model):
        assert isinstance(model, EventCompositionModel), \
            'model must be a {} instance'.format(
                get_class_name(EventCompositionModel))
        self.model = model
        self.set_embedding_model(model.word2vec)

    def log_evaluator_info(self):
        super(ArgumentCompositionEvaluator, self).log_evaluator_info()
        self.logger.info(
            'evaluator specific configs: use_max_score = {}'.format(
                self.use_max_score))

    def get_event_vector_list(self, event_input_list):
        assert all(isinstance(event_input, IndexedEvent) for event_input
                   in event_input_list), \
            'every event_input must be a {} instance'.format(
                get_class_name(IndexedEvent))

        num_event = len(event_input_list)
        pred_input = np.zeros(num_event, dtype=np.int32)
        subj_input = np.zeros(num_event, dtype=np.int32)
        obj_input = np.zeros(num_event, dtype=np.int32)
        pobj_input = np.zeros(num_event, dtype=np.int32)
        for event_idx, event_input in enumerate(event_input_list):
            pred_input[event_idx] = event_input.pred_input
            subj_input[event_idx] = event_input.subj_input
            obj_input[event_idx] = event_input.obj_input
            pobj_input[event_idx] = event_input.pobj_input

        projection_fn = self.model.event_vector_network.project
        event_vector_output = \
            projection_fn(pred_input, subj_input, obj_input, pobj_input)
        return list(event_vector_output)

    def evaluate_event_list(self, rich_event_list):
        pos_input_list = \
            [rich_event.get_pos_input(include_all_pobj=self.include_all_pobj)
                for rich_event in rich_event_list]
        pos_vector_list = self.get_event_vector_list(pos_input_list)

        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            context_vector_list = \
                pos_vector_list[:event_idx] + pos_vector_list[event_idx+1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=self.include_all_pobj, include_salience=False)

            for rich_arg, eval_input_list in eval_input_list_all:
                if not self.ignore_argument(rich_arg):
                    eval_vector_list = \
                        self.get_event_vector_list(eval_input_list)
                    most_coherent_idx = get_most_coherent(
                        eval_vector_list,
                        context_vector_list,
                        self.use_max_score
                    )
                    correct = (most_coherent_idx == rich_arg.get_target_idx())
                    num_choices = len(eval_input_list)
                    self.eval_stats.add_eval_result(
                        rich_arg.arg_type, correct, num_choices)
                    self.logger.debug(
                        'Processing {}, correct = {}, num_choices = {}'.format(
                            rich_arg.arg_type, correct, num_choices))
