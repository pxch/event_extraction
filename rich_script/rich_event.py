from event import Event
from indexed_input import SingleTrainingInput, SingleTrainingInputMultiPobj
from rich_argument import RichArgument
from util import Word2VecModel, get_class_name
from copy import deepcopy


class RichEvent(object):
    def __init__(self, pred_text, rich_subj, rich_obj, rich_pobj_list):
        self.pred_text = pred_text
        self.pred_idx = -1
        assert rich_subj is None or isinstance(rich_subj, RichArgument), \
            'rich_subj must be None or a {} instance'.format(
                get_class_name(RichArgument))
        self.rich_subj = rich_subj
        assert rich_obj is None or isinstance(rich_obj, RichArgument), \
            'rich_obj must be None or a {} instance'.format(
                get_class_name(RichArgument))
        self.rich_obj = rich_obj
        assert all(isinstance(rich_pobj, RichArgument) for rich_pobj
                   in rich_pobj_list), \
            'every rich_pobj must be a {} instance'.format(
                get_class_name(RichArgument))
        self.rich_pobj_list = rich_pobj_list
        # NOBUG: only set rich_pobj after call get_index()
        self.rich_pobj = None

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        if include_type:
            self.pred_idx = model.get_word_index(self.pred_text + '-PRED')
        else:
            self.pred_idx = model.get_word_index(self.pred_text)
        if self.rich_subj is not None:
            self.rich_subj.get_index(model, include_type=include_type)
        if self.rich_obj is not None:
            self.rich_obj.get_index(model, include_type=include_type)
        for rich_pobj in self.rich_pobj_list:
            rich_pobj.get_index(model, include_type=include_type)
        # select the first argument with indexed positive candidate and at least
        # one indexed negative candidate from rich_pobj_list as the rich_pobj
        for rich_pobj in self.rich_pobj_list:
            if rich_pobj.has_neg():
                self.rich_pobj = rich_pobj
                break

    def has_subj_neg(self):
        return self.rich_subj is not None and self.rich_subj.has_neg()

    def has_obj_neg(self):
        return self.rich_obj is not None and self.rich_obj.has_neg()

    def has_pobj_neg(self):
        return self.rich_pobj is not None and self.rich_pobj.has_neg()

    def has_neg(self, arg_type):
        assert arg_type in [0, 1, 2, 'SUBJ', 'OBJ', 'POBJ'], \
            'arg_type can only be 0/SUBJ, 1/OBJ, or 2/POBJ'
        if arg_type == 0 or arg_type == 'SUBJ':
            return self.has_subj_neg()
        elif arg_type == 1 or arg_type == 'OBJ':
            return self.has_obj_neg()
        elif arg_type == 2 or arg_type == 'POBJ':
            return self.has_pobj_neg()

    def get_word2vec_training_seq(self, include_all_pobj=True):
        sequence = [self.pred_text + '-PRED']
        all_arg_list = []
        if self.rich_subj is not None:
            all_arg_list.append(self.rich_subj)
        if self.rich_obj is not None:
            all_arg_list.append(self.rich_obj)
        if include_all_pobj:
            all_arg_list.extend(self.rich_pobj_list)
        else:
            if self.rich_pobj is not None:
                all_arg_list.append(self.rich_pobj)
        for arg in all_arg_list:
            sequence.append(arg.get_pos_text(include_type=True))
        return sequence

    def get_pos_training_input(self, include_all_pobj=True):
        if include_all_pobj:
            return SingleTrainingInputMultiPobj(
                self.pred_idx,
                self.rich_subj.get_pos_wv() if self.rich_subj else -1,
                self.rich_obj.get_pos_wv() if self.rich_obj else -1,
                [rich_pobj.get_pos_wv() for rich_pobj in self.rich_pobj_list]
            )
        else:
            return SingleTrainingInput(
                self.pred_idx,
                self.rich_subj.get_pos_wv() if self.rich_subj else -1,
                self.rich_obj.get_pos_wv() if self.rich_obj else -1,
                self.rich_pobj.get_pos_wv() if self.rich_pobj else -1
            )

    def get_neg_training_input(self, arg_type):
        assert arg_type in [0, 1, 2, 'SUBJ', 'OBJ', 'POBJ'], \
            'arg_type can only be 0/SUBJ, 1/OBJ, or 2/POBJ'
        pos_input = self.get_pos_training_input()
        neg_input_list = []
        if arg_type == 0 or arg_type == 'SUBJ':
            if self.has_subj_neg():
                for neg_idx in self.rich_subj.get_neg_wv_list():
                    neg_input = deepcopy(pos_input)
                    neg_input.set_subj(neg_idx)
                    neg_input_list.append(neg_input)
        elif arg_type == 1 or arg_type == 'OBJ':
            if self.has_obj_neg():
                for neg_idx in self.rich_obj.get_neg_wv_list():
                    neg_input = deepcopy(pos_input)
                    neg_input.set_obj(neg_idx)
                    neg_input_list.append(neg_input)
        elif arg_type == 2 or arg_type == 'POBJ':
            if self.has_pobj_neg():
                for neg_idx in self.rich_pobj.get_neg_wv_list():
                    neg_input = deepcopy(pos_input)
                    neg_input.set_pobj(neg_idx)
                    neg_input_list.append(neg_input)
        return neg_input_list

    def get_eval_input_list_all(self, include_all_pobj=True):
        results = []

        subj_eval_input_list = self.get_eval_input_list_subj(include_all_pobj)
        if subj_eval_input_list:
            results.append((self.rich_subj, subj_eval_input_list))

        obj_eval_input_list = self.get_eval_input_list_obj(include_all_pobj)
        if obj_eval_input_list:
            results.append((self.rich_obj, obj_eval_input_list))

        if include_all_pobj:
            for pobj_idx, rich_pobj in enumerate(self.rich_pobj_list):
                pobj_eval_input_list = \
                    self.get_eval_input_list_pobj_multi(pobj_idx)
                if pobj_eval_input_list:
                    results.append((rich_pobj, pobj_eval_input_list))
        else:
            pobj_eval_input_list = self.get_eval_input_list_pobj()
            if pobj_eval_input_list:
                results.append((self.rich_pobj, pobj_eval_input_list))

        return results

    def get_eval_input_list_subj(self, include_all_pobj=True):
        eval_input_list = []
        if self.rich_subj is not None and self.rich_subj.has_neg():
            pos_input = self.get_pos_training_input(
                include_all_pobj=include_all_pobj)
            for candidate_wv in self.rich_subj.candidate_wv_list:
                eval_input = deepcopy(pos_input)
                eval_input.set_subj(candidate_wv)
                eval_input_list.append(eval_input)
        return eval_input_list

    def get_eval_input_list_obj(self, include_all_pobj=True):
        eval_input_list = []
        if self.rich_obj is not None and self.rich_obj.has_neg():
            pos_input = self.get_pos_training_input(
                include_all_pobj=include_all_pobj)
            for candidate_wv in self.rich_obj.candidate_wv_list:
                eval_input = deepcopy(pos_input)
                eval_input.set_obj(candidate_wv)
                eval_input_list.append(eval_input)
        return eval_input_list

    def get_eval_input_list_pobj(self):
        eval_input_list = []
        if self.rich_pobj is not None and self.rich_pobj.has_neg():
            pos_input = self.get_pos_training_input(include_all_pobj=False)
            for candidate_wv in self.rich_pobj.candidate_wv_list:
                eval_input = deepcopy(pos_input)
                eval_input.set_pobj(candidate_wv)
                eval_input_list.append(eval_input)
        return eval_input_list

    def get_eval_input_list_pobj_multi(self, pobj_idx):
        assert 0 <= pobj_idx < len(self.rich_pobj_list)
        eval_input_list = []
        rich_pobj = self.rich_pobj_list[pobj_idx]
        if rich_pobj.has_neg():
            pos_input = self.get_pos_training_input(include_all_pobj=True)
            for candidate_wv in rich_pobj.candidate_wv_list:
                eval_input = deepcopy(pos_input)
                eval_input.set_pobj(pobj_idx, candidate_wv)
                eval_input_list.append(eval_input)
        return eval_input_list

    @classmethod
    def build(cls, event, entity_list, use_lemma=True, include_neg=True,
              include_prt=True, use_entity=True, use_ner=True,
              include_prep=True):
        assert isinstance(event, Event), 'event must be a {} instance'.format(
            get_class_name(Event))
        pred_text = event.pred.get_representation(
            use_lemma=use_lemma, include_neg=include_neg,
            include_prt=include_prt)
        rich_subj = None
        if event.subj is not None:
            rich_subj = RichArgument.build(
                'SUBJ', event.subj, entity_list, use_entity=use_entity,
                use_ner=use_ner, use_lemma=use_lemma)
        rich_obj = None
        if event.obj is not None:
            rich_obj = RichArgument.build(
                'OBJ', event.obj, entity_list, use_entity=use_entity,
                use_ner=use_ner, use_lemma=use_lemma)
        rich_pobj_list = []
        for prep, pobj in event.pobj_list:
            arg_type = 'PREP_' + prep if include_prep else 'PREP'
            rich_pobj = RichArgument.build(
                arg_type, pobj, entity_list, use_entity=use_entity,
                use_ner=use_ner, use_lemma=use_lemma)
            rich_pobj_list.append(rich_pobj)
        return cls(pred_text, rich_subj, rich_obj, rich_pobj_list)

    @classmethod
    def build_with_vocab_list(
            cls, event, pred_vocab_list, arg_vocab_list, ner_vocab_list,
            prep_vocab_list, entity_list, use_entity=True):
        assert isinstance(event, Event), 'event must be a {} instance'.format(
            get_class_name(Event))
        pred_text = event.pred.get_repr_universal(pred_vocab_list)
        rich_subj = None
        if event.subj is not None:
            rich_subj = RichArgument.build_with_vocab_list(
                'SUBJ', event.subj, arg_vocab_list, ner_vocab_list,
                entity_list, use_entity=use_entity)
        rich_obj = None
        if event.obj is not None:
            rich_obj = RichArgument.build_with_vocab_list(
                'OBJ', event.obj, arg_vocab_list, ner_vocab_list,
                entity_list, use_entity=use_entity)
        rich_pobj_list = []
        for prep, pobj in event.pobj_list:
            arg_type = 'PREP_' + prep if prep in prep_vocab_list else 'PREP'
            rich_pobj = RichArgument.build_with_vocab_list(
                arg_type, pobj, arg_vocab_list, ner_vocab_list,
                entity_list, use_entity=use_entity)
            rich_pobj_list.append(rich_pobj)
        return cls(pred_text, rich_subj, rich_obj, rich_pobj_list)
