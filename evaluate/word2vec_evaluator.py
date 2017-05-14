import numpy as np

from event_coherence_evaluator import EventCoherenceEvaluator
from rich_script import IndexedEvent, IndexedEventMultiPobj
from util import Word2VecModel, get_class_name, cos_sim


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


class Word2VecEvaluator(EventCoherenceEvaluator):
    def __init__(self, logger=None, ignore_first_mention=False, use_lemma=True,
                 include_type=True, use_max_score=True, include_all_pobj=True,
                 filter_stop_events=True):
        super(Word2VecEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention,
            use_lemma=use_lemma,
            include_type=include_type,
            use_max_score=use_max_score,
            include_all_pobj=include_all_pobj,
            filter_stop_events=filter_stop_events
        )

    def set_model(self, model):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.embedding_model = model
        self.model_name = self.embedding_model.name

    def get_event_vector(self, event_input, include_all_pobj=True):
        if include_all_pobj:
            assert isinstance(event_input, IndexedEventMultiPobj), \
                'event_input must be a {} instance ' \
                'when include_all_pobj=True'.format(
                    get_class_name(IndexedEventMultiPobj))
        else:
            assert isinstance(event_input, IndexedEvent), \
                'event_input must be a {} instance ' \
                'when include_all_pobj=False'.format(
                    get_class_name(IndexedEvent))
        # initialize event vector to be all zero
        vector = np.zeros(self.embedding_model.vector_size)
        # add vector for predicate
        pred_vector = self.embedding_model.get_index_vec(
            event_input.get_predicate())
        if pred_vector is not None:
            vector += pred_vector
        else:
            return None
        # add vectors for all arguments
        for arg_input in event_input.get_all_argument():
            arg_vector = self.embedding_model.get_index_vec(arg_input)
            if arg_vector is not None:
                vector += arg_vector
        return vector

    def evaluate_event(self, eval_input_list_all, context_input_list):
        context_vector_list = \
            [self.get_event_vector(
                context_input, include_all_pobj=self.include_all_pobj)
                for context_input in context_input_list]
        for rich_arg, eval_input_list in eval_input_list_all:
            if (not self.ignore_first_mention) or \
                    (not rich_arg.is_first_mention()):
                eval_vector_list = \
                    [self.get_event_vector(
                        eval_input, include_all_pobj=self.include_all_pobj)
                        for eval_input in eval_input_list]
                most_coherent_idx = get_most_coherent(
                    eval_vector_list,
                    context_vector_list,
                    self.use_max_score
                )
                correct = (most_coherent_idx == rich_arg.get_target_idx())
                num_choices = len(eval_input_list)
                self.eval_stats.add_eval_result(
                    rich_arg.arg_type,
                    correct,
                    num_choices
                )
                self.logger.debug(
                    'Processing {}, correct = {}, num_choices = {}'.format(
                        rich_arg.arg_type, correct, num_choices))
