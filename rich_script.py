from simple_script import Argument, Event, Script
from word2vec import Word2VecModel

from itertools import product
from random import sample


class SingleTrainingInput(object):
    def __init__(self, pred_idx, subj_idx, obj_idx, pobj_idx):
        self.pred_idx = pred_idx
        self.subj_idx = subj_idx
        self.obj_idx = obj_idx
        self.pobj_idx = pobj_idx

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return 'SingleTrainingInput: ' + self.to_text()

    def to_text(self):
        return '{},{},{},{}'.format(
            self.pred_idx, self.subj_idx, self.obj_idx, self.pobj_idx)

    @classmethod
    def from_text(cls, text):
        parts = text.strip().split(',')
        assert len(parts) == 4, \
            'expecting 4 parts separated by ",", found {}'.format(len(parts))
        pred_idx = int(parts[0])
        subj_idx = int(parts[1])
        obj_idx = int(parts[2])
        pobj_idx = int(parts[3])
        return cls(pred_idx, subj_idx, obj_idx, pobj_idx)


class PairTrainingInput(object):
    def __init__(self, left_input, pos_input, neg_input, arg_type):
        assert isinstance(left_input, SingleTrainingInput), \
            'left_input must be a SingleTrainingInput instance'
        self.left_input = left_input
        assert isinstance(pos_input, SingleTrainingInput), \
            'pos_input must be a SingleTrainingInput instance'
        self.pos_input = pos_input
        assert isinstance(neg_input, SingleTrainingInput), \
            'neg_input must be a SingleTrainingInput instance'
        self.neg_input = neg_input
        assert arg_type in [0, 1, 2], \
            'arg_type must be 0 (for subj), 1 (for obj), or 2(for pobj)'
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


class RichArgument(object):
    def __init__(self, arg_type, pos_text, neg_text_list):
        assert arg_type in ['SUBJ', 'OBJ'] or arg_type.startswith('PREP'), \
            'arg_type {} must be SUBJ/OBJ or starts with PREP'.format(arg_type)
        self.arg_type = arg_type
        self.pos_text = pos_text
        self.neg_text_list = neg_text_list
        self.has_neg = (len(self.neg_text_list) != 0)
        self.pos_idx = -1
        self.neg_idx_list = []

    @classmethod
    def build(cls, arg_type, arg, entity_list, use_entity=True, use_ner=True,
              use_lemma=True):
        assert arg is not None and isinstance(arg, Argument), \
            'arg must be an Argument instance, {} found'.format(type(arg))
        if arg.entity_idx != -1 and use_entity:
            assert 0 <= arg.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(arg.entity_idx)
            entity_text_list = [
                entity.get_representation(use_ner=use_ner, use_lemma=use_lemma)
                for entity in entity_list]
            pos_text = entity_text_list[arg.entity_idx]
            neg_text_list = \
                entity_text_list[:arg.entity_idx] + \
                entity_text_list[arg.entity_idx + 1:]
        else:
            pos_text = arg.get_representation(
                use_entity=False, entity_list=None,
                use_ner=use_ner, use_lemma=use_lemma)
            neg_text_list = []
        return cls(arg_type, pos_text, neg_text_list)

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a Word2VecModel instance'
        if include_type:
            self.pos_idx = model.get_word_index(
                self.pos_text + '-' + self.arg_type)
            self.neg_idx_list = [
                model.get_word_index(neg_text + '-' + self.arg_type)
                for neg_text in self.neg_text_list]
        else:
            self.pos_idx = model.get_word_index(self.pos_text)
            self.neg_idx_list = [
                model.get_word_index(neg_text)
                for neg_text in self.neg_text_list]


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
        assert arg_type in ['SUBJ', 'OBJ', 'POBJ', 'ALL'], \
            'arg_type must be among SUBJ/OBJ/POBJ/ALL'
        neg_input_list = []
        if arg_type == 'SUBJ' or arg_type == 'ALL':
            if self.has_subj_neg():
                for neg_idx in self.rich_subj.neg_idx_list:
                    neg_input = SingleTrainingInput(
                        self.pred_idx,
                        neg_idx,
                        self.rich_obj.pos_idx if self.rich_obj else -1,
                        self.rich_pobj.pos_idx if self.rich_pobj else -1
                    )
                    neg_input_list.append(neg_input)
        if arg_type == 'OBJ' or arg_type == 'ALL':
            if self.has_obj_neg():
                for neg_idx in self.rich_obj.neg_idx_list:
                    neg_input = SingleTrainingInput(
                        self.pred_idx,
                        self.rich_subj.pos_idx if self.rich_subj else -1,
                        neg_idx,
                        self.rich_pobj.pos_idx if self.rich_pobj else -1
                    )
                    neg_input_list.append(neg_input)
        if arg_type == 'POBJ' or arg_type == 'ALL':
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


class RichScript(object):
    def __init__(self, doc_name, rich_events, num_entities):
        self.doc_name = doc_name
        self.rich_events = rich_events
        self.num_events = len(self.rich_events)
        self.num_entities = num_entities

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a Word2VecModel instance'
        for rich_event in self.rich_events:
            rich_event.get_index(model, include_type=include_type)

    def get_word2vec_training_seq(self, include_all_pobj=True):
        sequence = []
        for rich_event in self.rich_events:
            sequence.extend(
                rich_event.get_word2vec_training_seq(include_all_pobj))
        return sequence

    def get_autoencoder_pretraining_input(self):
        results = []
        for rich_event in self.rich_events:
            results.append(rich_event.get_pos_training_input())
        return results

    def get_pair_training_input(self, num_pairs=1, num_neg_samples=1):
        # return empty list when number of entities is less than or equal to 1,
        # since there exists no negative samples
        if self.num_entities <= 1:
            return []
        results = []
        if num_pairs >= self.num_events:
            num_pairs = self.num_events - 1
        if num_neg_samples >= self.num_entities:
            num_neg_samples = self.num_entities - 1
        for idx, rich_event in enumerate(self.rich_events):
            # get positive right-hand event
            pos_event = rich_event.get_pos_training_input()
            # get list of left-hand event (equals to num_pairs)
            left_event_idx_list = sample(
                range(0, idx) + range(idx, self.num_events), num_pairs)
            left_event_list = [
                self.rich_events[left_idx].get_pos_training_input()
                for left_idx in left_event_idx_list]
            # get list of negative right-hand event for subject if possible
            if rich_event.has_subj_neg():
                neg_event_list = sample(
                    rich_event.get_neg_training_input('SUBJ'), num_neg_samples)
                for neg_event, left_event in product(
                        neg_event_list, left_event_list):
                    results.append(PairTrainingInput(
                        left_event, pos_event, neg_event, 0))
            if rich_event.has_obj_neg():
                neg_event_list = sample(
                    rich_event.get_neg_training_input('OBJ'), num_neg_samples)
                for neg_event, left_event in product(
                        neg_event_list, left_event_list):
                    results.append(PairTrainingInput(
                        left_event, pos_event, neg_event, 1))
            if rich_event.has_pobj_neg():
                neg_event_list = sample(
                    rich_event.get_neg_training_input('POBJ'), num_neg_samples)
                for neg_event, left_event in product(
                        neg_event_list, left_event_list):
                    results.append(PairTrainingInput(
                        left_event, pos_event, neg_event, 2))
        return results

    @classmethod
    def build(cls, script, use_lemma=True, include_neg=True, include_prt=True,
              use_entity=True, use_ner=True, include_prep=True):
        assert isinstance(script, Script), 'script must be a Script instance'
        rich_events = []
        if not script.has_entities():
            use_entity = False
        for event in script.events:
            rich_event = RichEvent.build(
                event,
                script.entities,
                use_lemma=use_lemma,
                include_neg=include_neg,
                include_prt=include_prt,
                use_entity=use_entity,
                use_ner=use_ner,
                include_prep=include_prep
            )
            rich_events.append(rich_event)
        return cls(script.doc_name, rich_events, len(script.entities))
