import numpy as np

from event_coherence_evaluator import EventCoherenceEvaluator
from rich_script import SingleTrainingInput, SingleTrainingInputMultiPobj
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
                 include_neg=True, include_prt=True, use_entity=True,
                 use_ner=True, include_prep=True, include_type=True,
                 use_max_score=True, include_all_pobj=True):
        super(Word2VecEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention,
            use_lemma=use_lemma,
            include_neg=include_neg,
            include_prt=include_prt,
            use_entity=use_entity,
            use_ner=use_ner,
            include_prep=include_prep,
            include_type=include_type,
            use_max_score=use_max_score,
            include_all_pobj=include_all_pobj
        )

    def set_model(self, model):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.embedding_model = model
        self.model_name = self.embedding_model.name

    def get_event_vector(self, event_input, include_all_pobj=True):
        if include_all_pobj:
            assert isinstance(event_input, SingleTrainingInputMultiPobj), \
                'event_input must be a {} instance ' \
                'when include_all_pobj=True'.format(
                    get_class_name(SingleTrainingInputMultiPobj))
        else:
            assert isinstance(event_input, SingleTrainingInput), \
                'event_input must be a {} instance ' \
                'when include_all_pobj=False'.format(
                    get_class_name(SingleTrainingInput))
        vector = np.zeros(self.embedding_model.vector_size)
        pred_vector = self.embedding_model.get_index_vec(event_input.pred_input)
        if pred_vector is not None:
            vector += pred_vector
        else:
            return None
        subj_vector = self.embedding_model.get_index_vec(event_input.subj_input)
        if subj_vector is not None:
            vector += subj_vector
        obj_vector = self.embedding_model.get_index_vec(event_input.obj_input)
        if obj_vector is not None:
            vector += obj_vector
        if include_all_pobj:
            for pobj_input in event_input.pobj_input_list:
                pobj_vector = self.embedding_model.get_index_vec(pobj_input)
                if pobj_vector is not None:
                    vector += pobj_vector
        else:
            pobj_vector = self.embedding_model.get_index_vec(
                event_input.pobj_input)
            if pobj_vector is not None:
                vector += pobj_vector
        return vector

    def evaluate_event(self, eval_input_list_all, context_input_list):
        context_vector_list = \
            [self.get_event_vector(
                context_input, include_all_pobj=self.include_all_pobj)
                for context_input in context_input_list]
        for rich_arg, eval_input_list in eval_input_list_all:
            if (not self.ignore_first_mention) or (not rich_arg.is_first_mention()):
                eval_vector_list = \
                    [self.get_event_vector(
                        eval_input, include_all_pobj=self.include_all_pobj)
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
                self.logger.debug(
                    'Processing {}, correct = {}, num_choices = {}'.format(
                        rich_arg.arg_type, correct, num_choices))
