import random

from indexed_input import PairTrainingInput
from rich_event import RichEvent
from script import Script
from util import Word2VecModel


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

    def get_pretraining_input(self):
        results = []
        for rich_event in self.rich_events:
            results.append(rich_event.get_pos_training_input())
        return results

    def get_pair_tuning_input(self, neg_type):
        # return empty list when number of entities is less than or equal to 1,
        # since there exists no negative samples
        if self.num_entities <= 1:
            return []
        assert neg_type in ['one', 'neg', 'all'], \
            'neg_type can only be ' \
            'one (one negative event and one left event), ' \
            'neg (one left event for every negative event), or ' \
            'all (every left event for every negative event)'
        results = []
        pos_input_list = [rich_event.get_pos_training_input()
                          for rich_event in self.rich_events]
        for pos_idx, pos_event in enumerate(self.rich_events):
            pos_input = pos_event.get_pos_training_input()
            left_input_idx_list = \
                range(0, pos_idx) + range(pos_idx, self.num_events)
            for arg in [0, 1, 2]:
                if pos_event.has_neg(arg):
                    if neg_type == 'one':
                        neg_input = random.choice(
                            pos_event.get_neg_training_input(arg))
                        left_input = pos_input_list[
                            random.choice(left_input_idx_list)]
                        results.append(PairTrainingInput(
                            left_input, pos_input, neg_input, arg))
                    else:
                        neg_input_list = pos_event.get_neg_training_input(arg)
                        for neg_input in neg_input_list:
                            if neg_type == 'neg':
                                left_input = pos_input_list[
                                    random.choice(left_input_idx_list)]
                                results.append(PairTrainingInput(
                                    left_input, pos_input, neg_input, arg))
                            else:
                                for left_input_idx in left_input_idx_list:
                                    left_input = pos_input_list[left_input_idx]
                                    results.append(PairTrainingInput(
                                        left_input, pos_input, neg_input, arg))
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
