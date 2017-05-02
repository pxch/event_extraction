from itertools import product

import document
from event import Event
from util import get_class_name


class EventScript:
    def __init__(self, doc_name):
        self.doc_name = doc_name
        self.events = []
        self.corefs = []

    def add_event(self, event):
        self.events.append(event)

    def add_coref(self, coref):
        self.corefs.append(coref)

    def sort(self):
        self.events = sorted(self.events, key=lambda e: e.idx_key())

    def pretty_print(self):
        results = '\nEvents\n'
        results += '\n'.join([event.print_with_idx() for event in self.events])
        results += '\n\nCandidates\n'
        results += '\n'.join([coref.pretty_print() for coref in self.corefs])
        return results

    def read_from_sentence(self, sent):
        assert isinstance(sent, document.Sentence), \
            'read_from_sentence must be called with a {} instance'.format(
                get_class_name(document.Sentence))
        for pred_token in sent.tokens:
            if pred_token.pos.startswith('VB'):
                # exclude "be" verbs
                if pred_token.lemma == 'be':
                    continue
                # FIXME: don't use predefined set of stop verbs
                # if pred_token.lemma in consts.STOP_VERBS: continue
                # TODO: exclude verbs in quotes
                # exclude modifying verbs
                if sent.dep_graph.lookup_label(
                        'gov', pred_token.token_idx, 'xcomp'):
                    continue

                neg = False
                if sent.dep_graph.lookup_label('gov', pred_token.token_idx, 'neg'):
                    neg = True

                subj_list = sent.get_subj_list(pred_token.token_idx)
                obj_list = sent.get_obj_list(pred_token.token_idx)
                pobj_list = sent.get_pobj_list(pred_token.token_idx)

                if (not subj_list) and (not obj_list):
                    continue
                if not subj_list:
                    subj_list.append(None)
                if not obj_list:
                    obj_list.append(None)

                for arg_tuple in product(subj_list, obj_list):
                    self.add_event(Event.construct(
                        pred_token,  # predicate
                        neg,  # negation flag
                        arg_tuple[0],  # subject
                        arg_tuple[1],  # object
                        pobj_list  # prepositional object list
                    ))

    @classmethod
    def construct(cls, doc):
        assert isinstance(doc, document.Document), \
            'read_from_document must be called with a {} instance'.format(
                get_class_name(document.Document))
        script = cls(doc.doc_name)
        for sent in doc.sents:
            script.read_from_sentence(sent)
        for coref in doc.corefs:
            script.add_coref(coref)
        script.sort()
        return script

    def get_training_seq(self):
        sequence = []
        for event in self.events:
            sequence.extend(event.get_training_seq())
        return sequence
