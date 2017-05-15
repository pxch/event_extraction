from copy import deepcopy

from event import Event
from indexed_event import IndexedEvent, IndexedEventMultiPobj
from rich_argument import BaseRichArgument
from rich_predicate import RichPredicate
from util import Word2VecModel, get_class_name


class RichEvent(object):
    def __init__(self, rich_pred, rich_subj, rich_obj, rich_pobj_list):
        assert isinstance(rich_pred, RichPredicate), \
            'rich_pred must be a {} instance'.format(
                get_class_name(RichPredicate))
        self.rich_pred = rich_pred
        assert rich_subj is None or isinstance(rich_subj, BaseRichArgument), \
            'rich_subj must be None or a {} instance'.format(
                get_class_name(BaseRichArgument))
        self.rich_subj = rich_subj
        assert rich_obj is None or isinstance(rich_obj, BaseRichArgument), \
            'rich_obj must be None or a {} instance'.format(
                get_class_name(BaseRichArgument))
        self.rich_obj = rich_obj
        assert all(isinstance(rich_pobj, BaseRichArgument) for rich_pobj
                   in rich_pobj_list), \
            'every rich_pobj must be a {} instance'.format(
                get_class_name(BaseRichArgument))
        self.rich_pobj_list = rich_pobj_list
        # NOBUG: only set rich_pobj after calling get_index()
        self.rich_pobj = None

    def get_index(self, model, include_type=True, use_unk=True,
                  pred_count_dict=None):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.rich_pred.get_index(
            model, include_type=include_type, use_unk=use_unk,
            pred_count_dict=pred_count_dict)
        if self.rich_subj is not None:
            self.rich_subj.get_index(
                model, include_type=include_type, use_unk=use_unk)
        if self.rich_obj is not None:
            self.rich_obj.get_index(
                model, include_type=include_type, use_unk=use_unk)
        for rich_pobj in self.rich_pobj_list:
            rich_pobj.get_index(
                model, include_type=include_type, use_unk=use_unk)
        # select the first argument with indexed positive candidate and at least
        # one indexed negative candidate from rich_pobj_list as the rich_pobj
        for rich_pobj in self.rich_pobj_list:
            if rich_pobj.has_neg():
                self.rich_pobj = rich_pobj
                break

    def get_arg_idx_list(self, include_all_pobj=False):
        # add arg_idx for rich_subj (1) and rich_obj (2)
        arg_idx_list = [1, 2]
        if include_all_pobj:
            # add arg_idx for all arguments in rich_pobj_list (4, 5, ...)
            for pobj_idx in range(len(self.rich_pobj_list)):
                arg_idx_list.append(4 + pobj_idx)
        else:
            # add arg_idx for rich_pobj (3)
            arg_idx_list.append(3)
        return arg_idx_list

    def get_argument(self, arg_idx):
        assert arg_idx in [1, 2, 3] or \
               (arg_idx - 4) in range(len(self.rich_pobj_list)), \
               'arg_idx can only be 1 (for rich_subj), 2 (for rich_obj), ' \
               '3 (for rich_pobj), or 4, 5, ... (for all arguments ' \
               'in rich_pobj_list)'
        if arg_idx == 1:
            return self.rich_subj
        elif arg_idx == 2:
            return self.rich_obj
        elif arg_idx == 3:
            return self.rich_pobj
        else:
            return self.rich_pobj_list[arg_idx - 4]

    def has_neg(self, arg_idx):
        argument = self.get_argument(arg_idx)
        return argument is not None and argument.has_neg()

    def get_word2vec_training_seq(
            self, pred_vocab_list, arg_vocab_list, ner_vocab_list,
            include_type=True, include_all_pobj=True):
        sequence = [self.rich_pred.get_text(pred_vocab_list=pred_vocab_list,
                                            include_type=include_type)]
        arg_idx_list = self.get_arg_idx_list(include_all_pobj=include_all_pobj)
        for arg_idx in arg_idx_list:
            argument = self.get_argument(arg_idx)
            if argument is not None:
                sequence.append(
                    argument.get_pos_text(arg_vocab_list=arg_vocab_list,
                                          ner_vocab_list=ner_vocab_list,
                                          include_type=include_type))
        return sequence

    def get_pos_input(self, include_all_pobj=False):
        # return None when the predicate is not indexed
        if self.rich_pred.get_wv() == -1:
            return None
        # TODO: remove support for include_all_pobj
        if include_all_pobj:
            return IndexedEventMultiPobj(
                self.rich_pred.get_wv(),
                self.rich_subj.get_pos_wv() if self.rich_subj else -1,
                self.rich_obj.get_pos_wv() if self.rich_obj else -1,
                [rich_pobj.get_pos_wv() for rich_pobj in self.rich_pobj_list]
            )
        else:
            return IndexedEvent(
                self.rich_pred.get_wv(),
                self.rich_subj.get_pos_wv() if self.rich_subj else -1,
                self.rich_obj.get_pos_wv() if self.rich_obj else -1,
                self.rich_pobj.get_pos_wv() if self.rich_pobj else -1
            )

    def get_neg_input_list(self, arg_idx, include_salience=True):
        # return empty list when the predicate is not indexed
        if self.rich_pred.get_wv() == -1:
            return []
        assert arg_idx in [1, 2, 3], \
            'arg_idx can only be 1 (for SUBJ), 2 (for OBJ) or 3 (for POBJ)'
        pos_input = self.get_pos_input(include_all_pobj=False)
        neg_input_list = []
        if self.has_neg(arg_idx):
            argument = self.get_argument(arg_idx)
            neg_wv_list = argument.get_neg_wv_list()
            neg_salience_list = argument.get_neg_salience_list()
            for arg_wv, arg_salience in zip(neg_wv_list, neg_salience_list):
                neg_input = deepcopy(pos_input)
                neg_input.set_argument(arg_idx, arg_wv)
                if include_salience:
                    neg_input_list.append((neg_input, arg_salience))
                else:
                    neg_input_list.append(neg_input)
        return neg_input_list

    def get_eval_input_list_all(self, include_all_pobj=True):
        # TODO: determine whether to include salience information as necessary
        # return empty list when the predicate is not indexed
        if self.rich_pred.get_wv() == -1:
            return []
        # TODO: remove support for include_all_pobj
        pos_input = self.get_pos_input(include_all_pobj=include_all_pobj)
        eval_input_list_all = []
        if pos_input is None:
            return eval_input_list_all
        arg_idx_list = self.get_arg_idx_list(include_all_pobj=include_all_pobj)
        for arg_idx in arg_idx_list:
            eval_input_list = []
            if self.has_neg(arg_idx):
                argument = self.get_argument(arg_idx)
                for arg_wv in argument.get_all_wv_list():
                    eval_input = deepcopy(pos_input)
                    eval_input.set_argument(arg_idx, arg_wv)
                    eval_input_list.append(eval_input)
                eval_input_list_all.append((argument, eval_input_list))
        return eval_input_list_all

    @classmethod
    def build(cls, event, rich_entity_list, prep_vocab_list, use_lemma=True):
        assert isinstance(event, Event), 'event must be a {} instance'.format(
            get_class_name(Event))
        rich_pred = RichPredicate.build(event.pred, use_lemma=use_lemma)
        rich_subj = None
        if event.subj is not None:
            rich_subj = BaseRichArgument.build(
                'SUBJ', event.subj, rich_entity_list, use_lemma=use_lemma)
        rich_obj = None
        if event.obj is not None:
            rich_obj = BaseRichArgument.build(
                'OBJ', event.obj, rich_entity_list, use_lemma=use_lemma)
        rich_pobj_list = []
        for prep, pobj in event.pobj_list:
            arg_type = 'PREP_' + prep if prep in prep_vocab_list else 'PREP'
            rich_pobj = BaseRichArgument.build(
                arg_type, pobj, rich_entity_list, use_lemma=use_lemma)
            rich_pobj_list.append(rich_pobj)
        return cls(rich_pred, rich_subj, rich_obj, rich_pobj_list)
