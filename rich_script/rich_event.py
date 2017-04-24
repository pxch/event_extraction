from event import Event
from indexed_input import SingleTrainingInput
from rich_argument import RichArgument
from word2vec import Word2VecModel


class RichEvent(object):
    def __init__(self, pred_text, rich_subj, rich_obj, rich_pobj_list):
        self.pred_text = pred_text
        self.pred_idx = -1
        assert rich_subj is None or isinstance(rich_subj, RichArgument), \
            'rich_subj must be None or a RichArgument instance'
        self.rich_subj = rich_subj
        assert rich_obj is None or isinstance(rich_obj, RichArgument), \
            'rich_obj must be None or a RichArgument instance'
        self.rich_obj = rich_obj
        assert all(isinstance(rich_pobj, RichArgument) for rich_pobj
                   in rich_pobj_list), \
            'every rich_pobj must be a RichArgument instance'
        self.rich_pobj_list = rich_pobj_list
        # select the first argument with entity linking from rich_pobj_list
        # as the rich_pobj
        self.rich_pobj = None
        for rich_pobj in self.rich_pobj_list:
            if rich_pobj.has_neg:
                self.rich_pobj = rich_pobj
                break

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a Word2VecModel instance'
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

    def has_subj_neg(self):
        return self.rich_subj is not None and self.rich_subj.has_neg

    def has_obj_neg(self):
        return self.rich_obj is not None and self.rich_obj.has_neg

    def has_pobj_neg(self):
        return self.rich_pobj is not None and self.rich_pobj.has_neg

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
            sequence.append(arg.pos_text + '-' + arg.arg_type)
        return sequence

    def get_pos_training_input(self):
        return SingleTrainingInput(
            self.pred_idx,
            self.rich_subj.pos_idx if self.rich_subj else -1,
            self.rich_obj.pos_idx if self.rich_obj else -1,
            self.rich_pobj.pos_idx if self.rich_pobj else -1
        )

    def get_neg_training_input(self, arg_type):
        assert arg_type in [0, 1, 2, 'SUBJ', 'OBJ', 'POBJ'], \
            'arg_type can only be 0/SUBJ, 1/OBJ, or 2/POBJ'
        neg_input_list = []
        if arg_type == 0 or arg_type == 'SUBJ':
            if self.has_subj_neg():
                for neg_idx in self.rich_subj.neg_idx_list:
                    neg_input = SingleTrainingInput(
                        self.pred_idx,
                        neg_idx,
                        self.rich_obj.pos_idx if self.rich_obj else -1,
                        self.rich_pobj.pos_idx if self.rich_pobj else -1
                    )
                    neg_input_list.append(neg_input)
        elif arg_type == 1 or arg_type == 'OBJ':
            if self.has_obj_neg():
                for neg_idx in self.rich_obj.neg_idx_list:
                    neg_input = SingleTrainingInput(
                        self.pred_idx,
                        self.rich_subj.pos_idx if self.rich_subj else -1,
                        neg_idx,
                        self.rich_pobj.pos_idx if self.rich_pobj else -1
                    )
                    neg_input_list.append(neg_input)
        elif arg_type == 2 or arg_type == 'POBJ':
            if self.has_pobj_neg():
                for neg_idx in self.rich_pobj.neg_idx_list:
                    neg_input = SingleTrainingInput(
                        self.pred_idx,
                        self.rich_subj.pos_idx if self.rich_subj else -1,
                        self.rich_obj.pos_idx if self.rich_obj else -1,
                        neg_idx
                    )
                    neg_input_list.append(neg_input)
        return neg_input_list

    @classmethod
    def build(cls, event, entity_list, use_lemma=True, include_neg=True,
              include_prt=True, use_entity=True, use_ner=True,
              include_prep=True):
        assert isinstance(event, Event), 'event must be an Event instance'
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

