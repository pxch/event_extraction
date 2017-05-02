import document
from util import consts, get_class_name


class Event:
    def __init__(self, pred, neg=False):
        assert pred is not None and isinstance(pred, document.Token), \
            'Predicate must be a {} instance'.format(
                get_class_name(document.Token))
        self.pred = pred
        self.neg = neg
        self.subj = None
        self.obj = None
        self.pobj_list = []

    @classmethod
    def construct(cls, pred, neg, subj, obj, pobj_list):
        event = cls(pred, neg)
        if subj is not None:
            event.add_subj(subj)
        if obj is not None:
            event.add_obj(obj)
        for prep, pobj in pobj_list:
            if prep != '' and pobj is not None:
                event.add_pobj(prep, pobj)
        return event

    def get_all_args(self):
        all_args = [('SUBJ', self.subj), ('OBJ', self.obj)] + \
                   [('PREP_' + prep, pobj) for prep, pobj in self.pobj_list]
        all_args = [arg for arg in all_args if arg[1] is not None]
        return all_args

    def get_all_args_with_coref(self):
        all_args = self.get_all_args()
        all_args = [arg for arg in all_args if arg[1].coref is not None]
        return all_args

    def idx_key(self):
        return '{:0>4d}{:0>4d}'.format(self.pred.sent_idx, self.pred.token_idx)

    def add_subj(self, subj):
        assert subj is not None and isinstance(subj, document.Token), \
            'Subject must be a {} instance'.format(
                get_class_name(document.Token))
        assert self.subj is None, 'Cannot add a subject while it already exists'
        self.subj = subj

    def add_obj(self, obj):
        assert obj is not None and isinstance(obj, document.Token), \
            'Object must be a {} instance'.format(
                get_class_name(document.Token))
        assert self.obj is None, 'Cannot add an object while it already exists'
        self.obj = obj

    def add_pobj(self, prep, pobj):
        assert prep != '', 'Cannot add a pobj with empty preposition!'
        assert pobj is not None and isinstance(pobj, document.Token), \
            'Prepositional object must be a {} instance'.format(
                get_class_name(document.Token))
        self.pobj_list.append((prep, pobj))

    def get_subj(self):
        return self.subj

    def get_obj(self):
        return self.obj

    def get_prep(self, pobj_idx):
        assert 0 <= pobj_idx < len(self.pobj_list), \
            'PObj index {} out of range'.format(pobj_idx)
        return self.pobj_list[pobj_idx][0]

    def get_pobj(self, pobj_idx):
        assert 0 <= pobj_idx < len(self.pobj_list), \
            'PObj index {} out of range'.format(pobj_idx)
        return self.pobj_list[pobj_idx][1]

    def subj_has_coref(self):
        return self.subj is not None and self.subj.coref is not None

    def obj_has_coref(self):
        return self.obj is not None and self.obj.coref is not None

    def pobj_has_coref(self, pobj_idx):
        return self.pobj_list[pobj_idx][1].coref is not None

    def get_pred_string_form(self):
        result = self.pred.string_form(use_lemma=True, include_compounds=True)
        if self.neg:
            result = 'not_' + result
        return result + '-PRED'

    def get_subj_string_form(self):
        if self.subj:
            result = self.subj.string_form(use_ner=True, use_lemma=True)
            if result != '':
                return result + '-SUBJ'
        return None

    def get_obj_string_form(self):
        if self.obj:
            result = self.obj.string_form(use_ner=True, use_lemma=True)
            if result != '':
                return result + '-OBJ'
        return None

    def get_pobj_string_form(self, pobj_idx, include_prep=True):
        if pobj_idx < 0 or pobj_idx >= len(self.pobj_list):
            return None
        result = self.pobj_list[pobj_idx][1].string_form(
            use_ner=True, use_lemma=True)
        prep = self.pobj_list[pobj_idx][0]
        if result != '':
            if include_prep and prep in consts.FREQ_PREPS:
                return result + '-PREP_' + prep
            else:
                return result + '-PREP'

    def get_training_seq(self, include_prep=True):
        sequence = [
            self.get_pred_string_form(),
            self.get_subj_string_form(),
            self.get_obj_string_form()
        ]
        for pobj_idx in range(len(self.pobj_list)):
            sequence.append(self.get_pobj_string_form(pobj_idx, include_prep))
        sequence = [token for token in sequence if token is not None]
        return sequence

    def __str__(self):
        result = '{} ( {}, {}'.format(
            str(self.pred),
            self.subj.print_with_coref() if self.subj else '_',
            self.obj.print_with_coref() if self.obj else '_')
        for prep, pobj in self.pobj_list:
            result += ', {}_{}'.format(
                prep,
                pobj.print_with_coref() if pobj else '_')
        result += ' )'
        return result

    def print_with_idx(self):
        return 'sent#{:0>3d}    {}'.format(
            self.pred.sent_idx, self.__str__())
