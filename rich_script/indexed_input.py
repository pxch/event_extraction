from copy import deepcopy

from util import get_class_name


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


class SingleTrainingInputMultiPobj(object):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input_list):
        self.pred_input = pred_input
        self.subj_input = subj_input
        self.obj_input = obj_input
        self.pobj_input_list = pobj_input_list

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'SingleTrainingInputMultiPobj: ' + self.to_text()

    def set_subj(self, subj_input):
        self.subj_input = subj_input

    def set_obj(self, obj_input):
        self.obj_input = obj_input

    def set_pobj(self, pobj_idx, pobj_input):
        assert 0 <= pobj_idx < len(self.pobj_input_list), \
            'pobj_idx {} out of range'.format(pobj_idx)
        self.pobj_input_list[pobj_idx] = pobj_input

    def to_text(self):
        return '{},{},{}{}'.format(
            self.pred_input, self.subj_input, self.obj_input,
            ',{}'.format(','.join(
                [pobj_input for pobj_input in self.pobj_input_list]))
            if self.pobj_input_list else '')

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) >= 3, \
            'expecting at least 3 parts separated by ",", found {}'.format(
                len(parts))
        pred_input = int(parts[0])
        subj_input = int(parts[1])
        obj_input = int(parts[2])
        pobj_input_list = []
        for p in parts[3:]:
            pobj_input_list.append(int(p))
        return cls(pred_input, subj_input, obj_input, pobj_input_list)


class PairTrainingInput(object):
    def __init__(self, left_input, pos_input, neg_input, arg_type):
        assert isinstance(left_input, SingleTrainingInput), \
            'left_input must be a {} instance'.format(
                get_class_name(SingleTrainingInput))
        self.left_input = deepcopy(left_input)
        assert isinstance(pos_input, SingleTrainingInput), \
            'pos_input must be a {} instance'.format(
                get_class_name(SingleTrainingInput))
        self.pos_input = deepcopy(pos_input)
        assert isinstance(neg_input, SingleTrainingInput), \
            'neg_input must be a {} instance'.format(
                get_class_name(SingleTrainingInput))
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
