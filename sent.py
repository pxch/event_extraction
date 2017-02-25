from dep import DepGraph
from event import Event
from itertools import product


class Sent:
    def __init__(self, idx):
        self.idx = idx
        self.tokens = []
        self.deps = []
        # dependency graph
        self.dep_graph = None
        # events
        self.events = []

    def add_token(self, token):
        token.add_idx_info(self.idx, len(self.tokens))
        self.tokens.append(token)

    def add_dep(self, dep):
        self.deps.append(dep)

    def __str__(self):
        return ' '.join([str(token) for token in self.tokens]) + '\t#DEP#\t' + \
               ' '.join([str(dep) for dep in self.deps])

    def pretty_print(self):
        return ' '.join([str(token) for token in self.tokens]) + '\n\t' + \
               ' '.join([str(dep) for dep in self.deps])

    def to_plain_text(self):
        return ' '.join([token.word for token in self.tokens])

    def build_dep_graph(self):
        self.dep_graph = DepGraph(self.idx, len(self.tokens))
        self.dep_graph.build(self.deps)

    def lookup_label(self, direction, token_idx, dep_label):
        return [self.tokens[idx] for idx in
                self.dep_graph.lookup_label(direction, token_idx, dep_label)]

    def process_verb_prt(self):
        for pred_idx in range(len(self.tokens)):
            pred_token = self.tokens[pred_idx]
            if pred_token.pos.startswith('VB'):
                if not pred_token.compounds_set:
                    pred_token.compounds.extend(self.lookup_label('gov', pred_idx, 'compound:prt'))
                    pred_token.compounds_set = True

    def get_head_token_idx(self, start_token_idx, end_token_idx):
        assert 0 <= start_token_idx < len(self.tokens), \
            'Start token idx {} out of range'.format(start_token_idx)
        assert 0 <= end_token_idx < len(self.tokens), \
            'End token idx {} out of range'.format(end_token_idx)
        assert start_token_idx <= end_token_idx, \
            'Start token idx {} larger than end token idx {}'.format(
                start_token_idx, end_token_idx)
        head_idx_map = []
        for token_idx in range(start_token_idx, end_token_idx + 1):
            head_trace = [token_idx]
            while start_token_idx <= head_trace[-1] <= end_token_idx:
                _, head_idx = self.dep_graph.get_parent(head_trace[-1])
                if head_idx in head_trace:
                    print 'Warning: In sentence #{}, token #{} has loop in its head trace list.'.format(
                        self.idx, token_idx)
                    break
                head_trace.append(head_idx)
            head_idx_map.append((token_idx, head_trace[-2]))
        head_idx_list = [head_idx for _, head_idx in head_idx_map]
        if min(head_idx_list) != max(head_idx_list):
            print 'Warning: In sentence #{}, tokens within the range [{}, {}] do not belong to the same subtree'.format(
                self.idx, start_token_idx, end_token_idx)
        if min(head_idx_list) < start_token_idx or min(head_idx_list) > end_token_idx:
            print 'Warning: In sentence #{}, head token index {} of [{}, {}] out of range!'.format(
                self.idx, min(head_idx_list), start_token_idx, end_token_idx)
        return min(head_idx_list)

    def get_subj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('gov', pred_idx, 'nsubj'))
        # agent of passive verb
        results.extend(self.lookup_label('gov', pred_idx, 'nmod:agent'))
        # controlling subject
        results.extend(self.lookup_label('gov', pred_idx, 'nsubj:xsubj'))
        return results

    def get_obj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('gov', pred_idx, 'dobj'))
        results.extend(self.lookup_label('gov', pred_idx, 'nsubjpass'))
        # TODO: whether or not to include acl relation
        # results.extend(self.lookup_label('dep', pred_idx, 'acl'))
        return results

    def get_pobj_list(self, pred_idx):
        results = []
        for label, indices in self.dep_graph.lookup('gov', pred_idx).items():
            if label.startswith('nmod') and ':' in label:
                prep_label = label.split(':')[1]
                if prep_label != 'agent':
                    results.extend([(prep_label, self.tokens[idx]) for idx in indices])
        return results

    def get_events_from_pred(self, pred_token):
        subj_list = self.get_subj_list(pred_token.token_idx)
        obj_list = self.get_obj_list(pred_token.token_idx)
        pobj_list = self.get_pobj_list(pred_token.token_idx)

        events = []
        if (not subj_list) and (not obj_list):
            return events
        if not subj_list:
            subj_list.append(None)
        if not obj_list:
            obj_list.append(None)

        for arg_tuple in product(subj_list, obj_list):
            events.append(Event.construct(
                pred_token,  # predicate
                arg_tuple[0],  # subject
                arg_tuple[1],  # direct object
                pobj_list  # prepositional object list
            ))
        return events

    def extract_events(self):
        event_list = []
        for pred_idx in range(len(self.tokens)):
            pred_token = self.tokens[pred_idx]
            if pred_token.pos.startswith('VB'):
                # exclude "be" verbs
                if pred_token.lemma == 'be':
                    continue
                # TODO: exclude stop verbs
                # TODO: exclude verbs in quotes
                # exclude modifying verbs
                if self.dep_graph.lookup_label('gov', pred_idx, 'xcomp'):
                    continue
                event_list.extend(self.get_events_from_pred(pred_token))
        # sort events based on predicate location
        self.events = sorted(event_list, key=lambda e: e.idx_key())

    def produce_token_list(self):
        surface_list = []
        lemma_list = []
        for idx, token in enumerate(self.tokens):
            if token.surface_output:
                surface_list.append((idx, token.surface_output))
            if token.lemma_output:
                lemma_list.append((idx, token.lemma_output))
        return surface_list, lemma_list

    def produce_syntax_list(self):
        surface_pair_list = []
        surface_triple_list = []
        lemma_pair_list = []
        lemma_triple_list = []
        for event in self.events:
            idx = event.pred.token_idx
            if event.pred.surface_output:
                surface_pair_list.append((idx, '{}-PRED'.format(
                    event.pred.surface_output)))
                if event.subj is not None and event.subj.surface_output:
                    surface_pair_list.append((idx, '{}-SUBJ'.format(
                        event.subj.surface_output)))
                    surface_triple_list.append((idx, '{}-SUBJ-{}'.format(
                        event.pred.surface_output, event.subj.surface_output)))
                if event.obj is not None and event.obj.surface_output:
                    surface_pair_list.append((idx, '{}-OBJ'.format(
                        event.obj.surface_output)))
                    surface_triple_list.append((idx, '{}-OBJ-{}'.format(
                        event.pred.surface_output, event.obj.surface_output)))
                for prep, pobj in event.pobj_list:
                    if pobj.surface_output:
                        surface_pair_list.append((idx, '{}-PREP_{}'.format(
                            pobj.surface_output, prep)))
                        surface_triple_list.append((idx, '{}-PREP_{}-{}'.format(
                            event.pred.surface_output, prep, pobj.surface_output)))
            if event.pred.lemma_output:
                lemma_pair_list.append((idx, '{}-PRED'.format(
                    event.pred.lemma_output)))
                if event.subj is not None and event.subj.lemma_output:
                    lemma_pair_list.append((idx, '{}-SUBJ'.format(
                        event.subj.lemma_output)))
                    lemma_triple_list.append((idx, '{}-SUBJ-{}'.format(
                        event.pred.lemma_output, event.subj.lemma_output)))
                if event.obj is not None and event.obj.lemma_output:
                    lemma_pair_list.append((idx, '{}-OBJ'.format(
                        event.obj.lemma_output)))
                    lemma_triple_list.append((idx, '{}-OBJ-{}'.format(
                        event.pred.lemma_output, event.obj.lemma_output)))
                for prep, pobj in event.pobj_list:
                    if pobj.lemma_output:
                        lemma_pair_list.append((idx, '{}-PREP_{}'.format(
                            pobj.lemma_output, prep)))
                        lemma_triple_list.append((idx, '{}-PREP_{}-{}'.format(
                            event.pred.lemma_output, prep, pobj.lemma_output)))
        return surface_pair_list, surface_triple_list, lemma_pair_list, lemma_triple_list
