from base_evaluator import BaseEvaluator
from event_embedding import EventEmbedding
import numpy as np


class MostSimEventEvaluator(BaseEvaluator):
    def __init__(self):
        BaseEvaluator.__init__(self)
        self.model = None
        self.head_only = False
        self.rep_only = False
        self.use_max_score = False

    def set_model(self, model):
        self.model = model

    def set_head_only(self, head_only):
        self.head_only = head_only

    def set_rep_only(self, rep_only):
        self.rep_only = rep_only

    def set_use_max_score(self, use_max_score):
        self.use_max_score = use_max_score

    def is_most_sim_event(self, arg_label, arg_coref_idx, corefs,
                          embedding_wo_arg, event_embeddings):
        sim_scores = []
        for coref in corefs:
            if self.model.syntax_label:
                coref_embedding = self.model.get_coref_embedding(
                    coref, '-' + arg_label, self.head_only, self.rep_only)
            else:
                coref_embedding = self.model.get_coref_embedding(
                    coref, '', self.head_only, self.rep_only)
            event_sim_scores = [self.cos_sim(
                embedding_wo_arg + coref_embedding, event_embedding)
                                for event_embedding in event_embeddings]
            if self.use_max_score:
                sim_scores.append(max(event_sim_scores))
            else:
                sim_scores.append(sum(event_sim_scores))
        most_sim_idx = sim_scores.index(max(sim_scores))
        return arg_coref_idx == most_sim_idx

    @staticmethod
    def cos_sim(vec1, vec2):
        if np.count_nonzero(vec1) == 0 or np.count_nonzero(vec2) == 0:
            return 0.0
        return vec1.dot(vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

    def evaluate_script(self, script):
        num_choices = len(script.corefs)

        event_embedding_list = []
        for event in script.events:
            event_embedding = EventEmbedding.construct(
                event, self.model, self.head_only, self.rep_only)
            if event_embedding is not None:
                event_embedding_list.append(event_embedding)

        all_event_embeddings = [event_embedding.get_embedding()
                                for event_embedding in event_embedding_list]

        for idx in range(len(event_embedding_list)):
            event_embedding = event_embedding_list[idx]
            other_event_embeddings = \
                all_event_embeddings[:idx] + all_event_embeddings[idx + 1:]

            # evaluate the subject argument if it is not None
            # and is pointed a coreference mention
            if event_embedding.get_subj_embedding() is not None:
                # label of the missing argument
                arg_label = 'SUBJ'
                # coreference index of the missing argument
                arg_coref_idx = event_embedding.event.subj.coref.idx
                # embedding of the target event without the missing argument
                embedding_wo_arg = event_embedding.get_embedding_wo_subj()
                self.eval_stats.add_eval_result(
                    arg_label,
                    self.is_most_sim_event(
                        arg_label, arg_coref_idx, script.corefs,
                        embedding_wo_arg, other_event_embeddings),
                    num_choices
                )

            # evaluate the object argument if it is not None
            # and is pointed a coreference mention
            if event_embedding.get_obj_embedding() is not None:
                # label of the missing argument
                arg_label = 'OBJ'
                # coreference index of the missing argument
                arg_coref_idx = event_embedding.event.obj.coref.idx
                # embedding of the target event without the missing argument
                embedding_wo_arg = event_embedding.get_embedding_wo_obj()
                self.eval_stats.add_eval_result(
                    arg_label,
                    self.is_most_sim_event(
                        arg_label, arg_coref_idx, script.corefs,
                        embedding_wo_arg, other_event_embeddings),
                    num_choices
                )

            for pobj_idx in range(len(event_embedding.event.pobj_list)):
                # evaluate the prepositional object argument
                # if it is pointed a coreference mention
                if event_embedding.get_pobj_embedding(pobj_idx) is not None:
                    # label of the missing argument
                    arg_label = \
                        'PREP_' + event_embedding.event.pobj_list[pobj_idx][0]
                    # coreference index of the missing argument
                    arg_coref_idx = \
                        event_embedding.event.pobj_list[pobj_idx][1].coref.idx
                    # embedding of the target event without the missing argument
                    embedding_wo_arg = \
                        event_embedding.get_embedding_wo_pobj(pobj_idx)
                    self.eval_stats.add_eval_result(
                        arg_label,
                        self.is_most_sim_event(
                            arg_label, arg_coref_idx, script.corefs,
                            embedding_wo_arg, other_event_embeddings),
                        num_choices
                    )
