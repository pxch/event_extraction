from numpy.linalg import norm
from numpy import count_nonzero


class EvalInstance:
    def __init__(self, arg_coref_idx, arg_label, pred_embedding, other_arg_embeddings):
        assert arg_label in ['SUBJ', 'OBJ'] or arg_label.startswith('PREP'), \
            'Argument label can only be SUBJ / OBJ or start with PREP'
        self.arg_coref_idx = arg_coref_idx
        self.arg_label = arg_label
        self.pred_embedding = pred_embedding
        self.other_arg_embeddings = other_arg_embeddings

    '''
    def eval_most_freq_coref(self, corefs):
      coref_freqs = [len(coref.mentions) for coref in corefs]
      coref_freqs[arg_coref_idx] -= 1
      most_freq_idx = coref_freqs.index(max(coref_freqs))
      return most_freq_idx == arg_coref_idx
    '''

    def eval_most_sim_arg(self, model, corefs,
                          use_other_args=False, syntax_suffix=False,
                          head_only=False, rep_only=False):
        target_embedding = model.zeros()
        target_embedding += self.pred_embedding
        if use_other_args:
            for embedding in self.other_arg_embeddings:
                target_embedding += embedding
        sim_scores = []
        for coref in corefs:
            if syntax_suffix:
                coref_embedding = coref.get_embedding(model, '-' + self.arg_label, head_only, rep_only)
            else:
                coref_embedding = coref.get_embedding(model, '', head_only, rep_only)
            sim_scores.append(cos_sim(target_embedding, coref_embedding))
        most_similar_idx = sim_scores.index(max(sim_scores))
        return most_similar_idx == self.arg_coref_idx

    def eval_most_sim_event(self, model, corefs, event_embedding_list,
                            use_max_score=False, syntax_suffix=False,
                            head_only=False, rep_only=False):
        target_embedding = model.zeros()
        target_embedding += self.pred_embedding
        for embedding in self.other_arg_embeddings:
            target_embedding += embedding
        sim_scores = []
        for coref in corefs:
            if syntax_suffix:
                coref_embedding = coref.get_embedding(model, '-' + self.arg_label, head_only, rep_only)
            else:
                coref_embedding = coref.get_embedding(model, '', head_only, rep_only)
            if use_max_score:
                sim_scores.append(max(
                    [cos_sim(target_embedding + coref_embedding, event_embedding)
                     for event_embedding in event_embedding_list]))
            else:
                sim_scores.append(sum(
                    [cos_sim(target_embedding + coref_embedding, event_embedding)
                     for event_embedding in event_embedding_list]))
        most_similar_idx = sim_scores.index(max(sim_scores))
        return most_similar_idx == self.arg_coref_idx


def cos_sim(vec1, vec2):
    if count_nonzero(vec1) == 0 or count_nonzero(vec2) == 0:
        return 0.0
    return vec1.dot(vec2) / norm(vec1) / norm(vec2)
