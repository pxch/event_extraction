import random

from indexed_event import IndexedEventTriple
from rich_event import RichEvent
from script import Script
from util import Word2VecModel, get_class_name


class RichScript(object):
    def __init__(self, doc_name, rich_events, num_entities):
        self.doc_name = doc_name
        self.rich_events = rich_events
        self.num_events = len(self.rich_events)
        self.num_entities = num_entities

    def get_index(self, model, include_type=True):
        assert isinstance(model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        for rich_event in self.rich_events:
            rich_event.get_index(model, include_type=include_type)

    # return list of events with indexed predicate (pred_idx != -1)
    def get_indexed_events(self):
        return [rich_event for rich_event in self.rich_events
                if rich_event.pred_idx != -1]

    def get_word2vec_training_seq(
            self, include_type=True, include_all_pobj=True):
        sequence = []
        for rich_event in self.rich_events:
            sequence.extend(
                rich_event.get_word2vec_training_seq(
                    include_type=include_type,
                    include_all_pobj=include_all_pobj))
        return sequence

    def get_pretraining_input_list(self):
        pretraining_input_list = []
        for rich_event in self.get_indexed_events():
            pos_input = rich_event.get_pos_input(include_all_pobj=False)
            if pos_input is not None:
                pretraining_input_list.append(pos_input)
        return pretraining_input_list

    def get_pair_tuning_input_list(self, neg_sample_type):
        # return empty list when number of entities is less than or equal to 1,
        # since there exists no negative inputs
        if self.num_entities <= 1:
            return []
        # return empty list when number of events with indexed predicate is
        # less than of equal to 1, since there exists no left inputs
        if len(self.get_indexed_events()) <= 1:
            return []
        assert neg_sample_type in ['one', 'neg', 'all'], \
            'neg_sample_type can only be ' \
            'one (one random negative event and one random left event), ' \
            'neg (one random left event for every negative event), or ' \
            'all (every left event for every negative event)'
        results = []
        pos_input_list = [rich_event.get_pos_input(include_all_pobj=False)
                          for rich_event in self.get_indexed_events()]
        for pos_idx, pos_event in enumerate(self.get_indexed_events()):
            pos_input = pos_input_list[pos_idx]
            if pos_input is None:
                continue
            left_input_idx_list = \
                range(0, pos_idx) + \
                range(pos_idx, len(self.get_indexed_events()))
            for arg_idx in [1, 2, 3]:
                if pos_event.has_neg(arg_idx):
                    if neg_sample_type == 'one':
                        neg_input = random.choice(
                            pos_event.get_neg_input_list(arg_idx))
                        left_input = pos_input_list[
                            random.choice(left_input_idx_list)]
                        results.append(
                            IndexedEventTriple(
                                left_input,
                                pos_input,
                                neg_input,
                                arg_idx
                            ))
                    else:
                        neg_input_list = pos_event.get_neg_input_list(arg_idx)
                        for neg_input in neg_input_list:
                            if neg_sample_type == 'neg':
                                left_input = pos_input_list[
                                    random.choice(left_input_idx_list)]
                                results.append(
                                    IndexedEventTriple(
                                        left_input,
                                        pos_input,
                                        neg_input,
                                        arg_idx
                                    ))
                            else:
                                for left_input_idx in left_input_idx_list:
                                    left_input = pos_input_list[left_input_idx]
                                    results.append(
                                        IndexedEventTriple(
                                            left_input,
                                            pos_input,
                                            neg_input,
                                            arg_idx
                                        ))
        return results

    @classmethod
    def build(cls, script, use_lemma=True, include_neg=True, include_prt=True,
              use_entity=True, use_ner=True, include_prep=True):
        assert isinstance(script, Script), \
            'script must be a {} instance'.format(get_class_name(Script))
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

    @classmethod
    def build_with_vocab_list(cls, script, pred_vocab_list, arg_vocab_list,
                              ner_vocab_list, prep_vocab_list, use_entity=True):
        assert isinstance(script, Script), \
            'script must be a {} instance'.format(get_class_name(Script))
        rich_events = []
        if not script.has_entities():
            use_entity = False
        for event in script.events:
            rich_event = RichEvent.build_with_vocab_list(
                event,
                pred_vocab_list=pred_vocab_list,
                arg_vocab_list=arg_vocab_list,
                ner_vocab_list=ner_vocab_list,
                prep_vocab_list=prep_vocab_list,
                entity_list=script.entities,
                use_entity=use_entity,
            )
            rich_events.append(rich_event)
        return cls(script.doc_name, rich_events, len(script.entities))
