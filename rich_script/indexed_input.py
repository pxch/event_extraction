from copy import deepcopy


class SingleTrainingInput(object):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input):
        self.pred_input = pred_input
        self.subj_input = subj_input
        self.obj_input = obj_input
        self.pobj_input = pobj_input

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'SingleTrainingInput: ' + self.to_text()

    def to_text(self):
        return '{},{},{},{}'.format(
            self.pred_input, self.subj_input, self.obj_input, self.pobj_input)

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) == 4, \
            'expecting 4 parts separated by ",", found {}'.format(len(parts))
        pred_input = int(parts[0])
        subj_input = int(parts[1])
        obj_input = int(parts[2])
        pobj_input = int(parts[3])
        return cls(pred_input, subj_input, obj_input, pobj_input)


class PairTrainingInput(object):
    def __init__(self, left_input, pos_input, neg_input, arg_type):
        assert isinstance(left_input, SingleTrainingInput), \
            'left_input must be a SingleTrainingInput instance'
        self.left_input = deepcopy(left_input)
        assert isinstance(pos_input, SingleTrainingInput), \
            'pos_input must be a SingleTrainingInput instance'
        self.pos_input = deepcopy(pos_input)
        assert isinstance(neg_input, SingleTrainingInput), \
            'neg_input must be a SingleTrainingInput instance'
        self.neg_input = deepcopy(neg_input)
        assert arg_type in [0, 1, 2], \
            'arg_type must be 0 (for subj), 1 (for obj), or 2 (for pobj)'
        self.arg_type = arg_type

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'PairTrainingInput: ' + self.to_text()

    def to_text(self):
        return ' / '.join([self.left_input.to_text(), self.pos_input.to_text(),
                           self.neg_input.to_text(), str(self.arg_type)])

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(' / ')
        assert len(parts) == 4, \
            'expecting 4 parts separated by " / ", found {}'.format(len(parts))
        left_input = SingleTrainingInput.from_text(parts[0])
        pos_input = SingleTrainingInput.from_text(parts[1])
        neg_input = SingleTrainingInput.from_text(parts[2])
        arg_type = int(parts[3])
        return cls(left_input, pos_input, neg_input, arg_type)
