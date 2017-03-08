# TODO: whether or not to include negation information in event representation
class Event:
    def __init__(self, pred):
        assert pred is not None and pred.__class__.__name__ == 'Token', \
            'Predicate must be an instance of Token'
        self.pred = pred
        self.subj = None
        self.obj = None
        self.pobj_list = []
        self.neg = False

    @classmethod
    def construct(cls, pred, subj, obj, pobj_list):
        event = cls(pred)
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
        assert subj is not None and subj.__class__.__name__ == 'Token', \
            'Subject must be an instance of Token'
        assert self.subj is None, 'Cannot add a subject while it already exists'
        self.subj = subj

    def add_obj(self, obj):
        assert obj is not None and obj.__class__.__name__ == 'Token', \
            'Object must be an instance of Token'
        assert self.obj is None, 'Cannot add an object while it already exists'
        self.obj = obj

    def add_pobj(self, prep, pobj):
        assert prep != '', 'Cannot add a pobj with empty preposition!'
        assert pobj is not None and pobj.__class__.__name__ == 'Token', \
            'Prepositional object must be an instance of Token'
        self.pobj_list.append((prep, pobj))

    def get_subj(self):
        return self.subj

    def get_obj(self):
        return self.obj

    def get_pobj(self, pobj_idx):
        assert 0 <= pobj_idx < len(self.pobj_list), \
            'PObj index {} out of range'.format(pobj_idx)
        return self.pobj_list[pobj_idx]

    def subj_has_coref(self):
        return self.subj is not None and self.subj.coref is not None

    def obj_has_coref(self):
        return self.obj is not None and self.obj.coref is not None

    def pobj_has_coref(self, pobj_idx):
        return self.pobj_list[pobj_idx][1].coref is not None

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
