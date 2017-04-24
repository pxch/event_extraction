import event_script
import re
from token import Predicate, Argument
from warnings import warn


class Event(object):
    def __init__(self, pred, subj, obj, pobj_list):
        if not isinstance(pred, Predicate):
            raise ParseEventError('pred must be a Predicate instance')
        self.pred = pred
        if not (subj is None or isinstance(subj, Argument)):
            raise ParseEventError('subj must be None or an Argument instance')
        self.subj = subj
        if not (obj is None or isinstance(obj, Argument)):
            raise ParseEventError('obj must be None or an Argument instance')
        self.obj = obj
        if not all(prep != '' for prep, _ in pobj_list):
            warn('some of prep(s) in pobj_list are empty')
        if not all(isinstance(pobj, Argument) for _, pobj in pobj_list):
            raise ParseEventError('every pobj must be an Argument instance')
        self.pobj_list = pobj_list
        self.pred_text = ''
        self.subj_text = ''
        self.obj_text = ''
        self.pobj_text_list = []

    def __eq__(self, other):
        return self.pred == other.pred and self.subj == other.subj and \
               self.obj == other.obj and \
               all(prep == other_prep and pobj == other_pobj
                   for (prep, pobj), (other_prep, other_pobj)
                   in zip(self.pobj_list, other.pobj_list))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_all_args(self):
        all_args = [self.subj, self.obj] + [pobj for _, pobj in self.pobj_list]
        all_args = [arg for arg in all_args if arg is not None]
        return all_args

    def to_text(self):
        return '{} :SUBJ: {} :OBJ: {}{}'.format(
            self.pred.to_text(),
            self.subj.to_text() if self.subj is not None else 'NONE',
            self.obj.to_text() if self.obj is not None else 'NONE',
            ''.join([' :POBJ: {} : {}'.format(prep, pobj.to_text())
                     for prep, pobj in self.pobj_list])
        )

    event_re = re.compile(
        r'^(?P<pred>[^:]*) :SUBJ: (?P<subj>[^:]*) :OBJ: (?P<obj>[^:]*)'
        r'((?: :POBJ: )(?P<prep>[^:]*)(?: : )(?P<pobj>[^:]*))*$')

    @classmethod
    def from_text(cls, text):
        parts = [p for p in re.split(' :(?:SUBJ|OBJ|POBJ): ', text)]
        if len(parts) < 3:
            raise ParseEventError('expected at least 3 parts, separated by ::, '
                                  'got {}: {}'.format(len(parts), text))
        pred = Predicate.from_text(parts[0])
        subj = None
        if parts[1] != 'NONE':
            subj = Argument.from_text(parts[1])
        obj = None
        if parts[2] != 'NONE':
            obj = Argument.from_text(parts[2])
        pobj_list = []
        if len(parts) > 3:
            for part in parts[3:]:
                prep, pobj = part.split(' : ')
                if prep != '':
                    pobj_list.append((prep, Argument.from_text(pobj)))
        return cls(pred, subj, obj, pobj_list)

    @classmethod
    def from_event(cls, event):
        if not isinstance(event, event_script.Event):
            raise ParseEventError(
                'from_event must be called with a {}.{} instance'.format(
                    event_script.Event.__module__, event_script.Event.__name__))
        return cls.from_tokens(event.pred, event.neg, event.subj, event.obj,
                               event.pobj_list)

    @classmethod
    def from_tokens(cls, pred_token, neg, subj_token, obj_token,
                    pobj_token_list):
        pred = Predicate.from_token(pred_token, neg=neg)
        subj = Argument.from_token(subj_token) \
            if subj_token is not None else None
        obj = Argument.from_token(obj_token) if obj_token is not None else None
        pobj_list = [(prep, Argument.from_token(pobj)) for prep, pobj
                     in pobj_token_list]
        return cls(pred, subj, obj, pobj_list)


class ParseEventError(Exception):
    pass
