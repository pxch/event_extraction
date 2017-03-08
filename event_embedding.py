import numpy as np
from warnings import warn


class EventEmbedding:
    def __init__(self, event, dimension):
        self.event = event
        self.dimension = dimension
        self.embedding = np.zeros(dimension)
        self.pred_embedding = None
        self.subj_embedding = None
        self.obj_embedding = None
        self.pobj_embedding_list = []

    def set_pred_embedding(self, pred_embedding):
        assert pred_embedding is not None, 'Predicate embedding cannot be None'
        assert type(pred_embedding) == np.ndarray, \
            'Embedding must be of type numpy.ndarray'

        if self.pred_embedding is not None:
            warn('Overriding existing pred_embedding')
        self.pred_embedding = pred_embedding
        self.embedding += pred_embedding

    def set_subj_embedding(self, subj_embedding):
        assert subj_embedding is None or \
               self.embedding.size == subj_embedding.size, 'Dimension mismatch'
        self.subj_embedding = subj_embedding
        if subj_embedding is not None:
            self.embedding += subj_embedding

    def set_obj_embedding(self, obj_embedding):
        assert obj_embedding is None or \
               self.embedding.size == obj_embedding.size, 'Dimension mismatch'
        self.obj_embedding = obj_embedding
        if obj_embedding is not None:
            self.embedding += obj_embedding

    def add_pobj_embedding(self, pobj_embedding):
        assert pobj_embedding is None or \
               self.embedding.size == pobj_embedding.size, 'Dimension mismatch'
        self.pobj_embedding_list.append(pobj_embedding)
        if pobj_embedding is not None:
            self.embedding += pobj_embedding

    def get_subj_embedding(self):
        return self.subj_embedding

    def get_obj_embedding(self):
        return self.obj_embedding

    def get_pobj_embedding(self, pobj_idx):
        assert 0 <= pobj_idx < len(self.pobj_embedding_list), \
            'PObj index {} out of range'.format(pobj_idx)
        return self.pobj_embedding_list[pobj_idx]

    def get_embedding(self):
        return self.embedding

    def get_embedding_wo_subj(self):
        if self.subj_embedding is not None:
            return self.embedding - self.subj_embedding
        else:
            return self.embedding

    def get_embedding_wo_obj(self):
        if self.obj_embedding is not None:
            return self.embedding - self.obj_embedding
        else:
            return self.embedding

    def get_embedding_wo_pobj(self, pobj_idx):

        assert 0 <= pobj_idx < len(self.pobj_embedding_list), \
            'PObj index {} out of range'.format(pobj_idx)
        if self.pobj_embedding_list[pobj_idx] is not None:
            return self.embedding - self.pobj_embedding_list[pobj_idx]
        else:
            return self.embedding

    @classmethod
    def construct(cls, event, model, head_only=False, rep_only=False):

        event_embedding = cls(event, model.dimension)

        # set predicate embedding
        if model.syntax_label:
            pred_embedding = model.get_token_embedding(event.pred, '-PRED')
        else:
            pred_embedding = model.get_token_embedding(event.pred)
        if pred_embedding is not None:
            event_embedding.set_pred_embedding(pred_embedding)
        # return None if event.predicate is out of vocabulary
        else:
            return None

        # set subject embedding
        if event.subj is None or event.subj.coref is None:
            subj_embedding = None
        else:
            if model.syntax_label:
                subj_embedding = model.get_coref_embedding(
                    event.subj.coref, '-SUBJ', head_only, rep_only)
            else:
                subj_embedding = model.get_coref_embedding(
                    event.subj.coref, '', head_only, rep_only)
        event_embedding.set_subj_embedding(subj_embedding)

        # set object embedding
        if event.obj is None or event.obj.coref is None:
            obj_embedding = None
        else:
            if model.syntax_label:
                obj_embedding = model.get_coref_embedding(
                    event.obj.coref, '-OBJ', head_only, rep_only)
            else:
                obj_embedding = model.get_coref_embedding(
                    event.obj.coref, '', head_only, rep_only)
        event_embedding.set_obj_embedding(obj_embedding)

        # set list of pobj embeddings
        for prep, pobj in event.pobj_list:
            if pobj.coref is None:
                pobj_embedding = None
            else:
                if model.syntax_label:
                    pobj_embedding = model.get_coref_embedding(
                        pobj.coref, '-PREP_' + prep, head_only, rep_only)
                else:
                    pobj_embedding = model.get_coref_embedding(
                        pobj.coref, '', head_only, rep_only)
            event_embedding.add_pobj_embedding(pobj_embedding)

        return event_embedding
