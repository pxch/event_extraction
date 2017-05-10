import abc
from copy import deepcopy

from util import get_class_name


class BaseIndexedEvent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, pred_input, subj_input, obj_input):
        # FIXME
        # assert pred_input != -1, 'predicate input cannot be -1 (non indexed)'
        self.pred_input = pred_input
        self.subj_input = subj_input
        self.obj_input = obj_input

    def __str__(self):
        return self.to_text()

    @abc.abstractmethod
    def __repr__(self):
        return

    @abc.abstractmethod
    def to_text(self):
        return

    def get_predicate(self):
        return self.pred_input

    @abc.abstractmethod
    def get_all_argument(self):
        return

    @abc.abstractmethod
    def get_argument(self, arg_idx):
        return

    @abc.abstractmethod
    def set_argument(self, arg_idx, arg_input):
        return


class IndexedEvent(BaseIndexedEvent):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input):
        super(IndexedEvent, self).__init__(pred_input, subj_input, obj_input)
        self.pobj_input = pobj_input

    def __repr__(self):
        return 'Indexed Event: ' + self.to_text()

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

    def get_all_argument(self):
        return [self.subj_input, self.obj_input, self.pobj_input]

    def get_argument(self, arg_idx):
        assert arg_idx in [1, 2, 3], \
            'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
            'or 3 (for pobj_input)'
        if arg_idx == 1:
            return self.subj_input
        elif arg_idx == 2:
            return self.obj_input
        else:
            return self.pobj_input

    def set_argument(self, arg_idx, arg_input):
        assert arg_idx in [1, 2, 3], \
            'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
            'or 3 (for pobj_input)'
        if arg_idx == 1:
            self.subj_input = arg_input
        elif arg_idx == 2:
            self.obj_input = arg_input
        else:
            self.pobj_input = arg_input


class IndexedEventMultiPobj(BaseIndexedEvent):
    def __init__(self, pred_input, subj_input, obj_input, pobj_input_list):
        super(IndexedEventMultiPobj, self).__init__(
            pred_input, subj_input, obj_input)
        self.pobj_input_list = pobj_input_list

    def __repr__(self):
        return 'Indexed Event (Multiple Pobj): ' + self.to_text()

    def to_text(self):
        return '{},{},{}{}'.format(
            self.pred_input, self.subj_input, self.obj_input,
            ',{}'.format(','.join(
                [str(pobj_input) for pobj_input in self.pobj_input_list]))
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

    def get_all_argument(self):
        return [self.subj_input, self.obj_input] + self.pobj_input_list

    def get_argument(self, arg_idx):
        # skip arg_idx 3 to avoid confusion with IndexedEvent
        assert arg_idx in [1, 2] or \
               (arg_idx - 4) in range(len(self.pobj_input_list)), \
               'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
               'or 4, 5, ... (for all arguments in pobj_input_list)'
        if arg_idx == 1:
            return self.subj_input
        elif arg_idx == 2:
            return self.obj_input
        else:
            return self.pobj_input_list[arg_idx - 4]

    def set_argument(self, arg_idx, arg_input):
        # skip arg_idx 3 to avoid confusion with IndexedEvent
        assert arg_idx in [1, 2] or \
               (arg_idx - 4) in range(len(self.pobj_input_list)), \
               'arg_idx can only be 1 (for subj_input), 2 (for obj_input), ' \
               'or 4, 5, ... (for all arguments in pobj_input_list)'
        if arg_idx == 1:
            self.subj_input = arg_input
        elif arg_idx == 2:
            self.obj_input = arg_input
        else:
            self.pobj_input_list[arg_idx - 4] = arg_input


class IndexedEventTriple(object):
    def __init__(self, left_list, pos_list, neg_list, arg_idx):
        assert isinstance(left_list, IndexedEvent), \
            'left_list must be a {} instance'.format(
                get_class_name(IndexedEvent))
        self.left_list = deepcopy(left_list)
        assert isinstance(pos_list, IndexedEvent), \
            'pos_input must be a {} instance'.format(
                get_class_name(IndexedEvent))
        self.pos_list = deepcopy(pos_list)
        assert isinstance(neg_list, IndexedEvent), \
            'neg_input must be a {} instance'.format(
                get_class_name(IndexedEvent))
        self.neg_list = deepcopy(neg_list)
        assert arg_idx in [1, 2, 3], \
            'arg_type must be 1 (for subj), 2 (for obj), or 3 (for pobj)'
        self.arg_idx = arg_idx
        # TODO: add extra features for entity saliency
        self.saliency_features = None

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'Indexed Event Triple: ' + self.to_text()

    def to_text(self):
        return ' / '.join([self.left_list.to_text(), self.pos_list.to_text(),
                           self.neg_list.to_text(), str(self.arg_idx)])

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(' / ')
        assert len(parts) == 4, \
            'expecting 4 parts separated by " / ", found {}'.format(len(parts))
        left_list = IndexedEvent.from_text(parts[0])
        pos_list = IndexedEvent.from_text(parts[1])
        neg_list = IndexedEvent.from_text(parts[2])
        arg_idx = int(parts[3])
        return cls(left_list, pos_list, neg_list, arg_idx)
