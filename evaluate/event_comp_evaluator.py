import numpy as np

from event_coherence_evaluator import EventCoherenceEvaluator
from event_composition import EventCompositionModel
from rich_script import IndexedEvent
from util import get_class_name


class EventCompositionEvaluator(EventCoherenceEvaluator):
    def __init__(self, logger=None, ignore_first_mention=False,
                 use_lemma=True, include_type=True, use_max_score=True,
                 filter_stop_events=True):
        super(EventCompositionEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention,
            use_lemma=use_lemma,
            include_type=include_type,
            use_max_score=use_max_score,
            include_all_pobj=False,
            filter_stop_events=filter_stop_events
        )

    def set_model(self, model):
        assert isinstance(model, EventCompositionModel), \
            'model must be a {} instance'.format(
                get_class_name(EventCompositionModel))
        self.model = model
        self.embedding_model = self.model.arg_comp_model.word2vec
        self.model_name = 'event_comp with ' + self.embedding_model.name

    def get_most_coherent(self, arg_type, eval_input_list, context_input_list,
                          use_max_score=True):
        coherence_fn = self.model.coherence_fn

        coherence_score_list = []
        num_context = len(context_input_list)

        if arg_type == 'SUBJ':
            arg_type_input = np.asarray([1] * num_context).astype(np.int32)
        elif arg_type == 'OBJ':
            arg_type_input = np.asarray([2] * num_context).astype(np.int32)
        elif arg_type.startswith('PREP'):
            arg_type_input = np.asarray([3] * num_context).astype(np.int32)
        else:
            raise ValueError(
                'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(
                    arg_type))

        pred_input_a = np.zeros(num_context, dtype=np.int32)
        subj_input_a = np.zeros(num_context, dtype=np.int32)
        obj_input_a = np.zeros(num_context, dtype=np.int32)
        pobj_input_a = np.zeros(num_context, dtype=np.int32)
        for context_idx, context_input in enumerate(context_input_list):
            assert isinstance(context_input, IndexedEvent), \
                'context_input must be a {} instance'.format(
                    get_class_name(IndexedEvent))
            pred_input_a[context_idx] = context_input.pred_input
            subj_input_a[context_idx] = context_input.subj_input
            obj_input_a[context_idx] = context_input.obj_input
            pobj_input_a[context_idx] = context_input.pobj_input

        for eval_input in eval_input_list:
            assert isinstance(eval_input, IndexedEvent), \
                'eval_input must be a {} instance'.format(
                    get_class_name(IndexedEvent))
            pred_input_b = np.asarray(
                [eval_input.pred_input] * num_context).astype(np.int32)
            subj_input_b = np.asarray(
                [eval_input.subj_input] * num_context).astype(np.int32)
            obj_input_b = np.asarray(
                [eval_input.obj_input] * num_context).astype(np.int32)
            pobj_input_b = np.asarray(
                [eval_input.pobj_input] * num_context).astype(np.int32)

            coherence_output = coherence_fn(
                pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
                pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
                arg_type_input)
            if use_max_score:
                coherence_score_list.append(coherence_output.max())
            else:
                coherence_score_list.append(coherence_output.sum())

        most_coherent_idx = coherence_score_list.index(
            max(coherence_score_list))
        return most_coherent_idx

    def evaluate_event(self, eval_input_list_all, context_input_list):
        for rich_arg, eval_input_list in eval_input_list_all:
            if (not self.ignore_first_mention) or \
                    (not rich_arg.is_first_mention()):
                most_coherent_idx = self.get_most_coherent(
                    rich_arg.arg_type, eval_input_list, context_input_list,
                    self.use_max_score)
                correct = (most_coherent_idx == rich_arg.target_idx)
                num_choices = len(eval_input_list)
                self.eval_stats.add_eval_result(
                    rich_arg.arg_type,
                    correct,
                    num_choices
                )
                self.logger.debug(
                    'Processing {}, correct = {}, num_choices = {}'.format(
                        rich_arg.arg_type, correct, num_choices))
