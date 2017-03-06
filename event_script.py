from event import Event
from itertools import product


class EvalStats:
    class AccuracyStats:
        def __init__(self):
            self.num_cases = 0
            self.num_positives = 0
            self.num_choices = []

        def add_eval_result(self, correct, num_choices):
            self.num_cases += 1
            self.num_positives += correct
            self.num_choices.append(num_choices)
            # self.check_consistency()

        def check_consistency(self):
            assert len(self.num_choices) == self.num_cases, \
                'Length of num_choices not equal to number of cases.'
            assert 0 <= self.num_positives <= self.num_cases, \
                'Number of positive cases must be positive and ' \
                'smaller than number of cases'

        def add_accuracy_stats(self, accuracy_stats):
            self.num_cases += accuracy_stats.num_cases
            self.num_positives += accuracy_stats.num_positives
            self.num_choices.extend(accuracy_stats.num_choices)
            # self.check_consistency()

        def __str__(self):
            return '{}/{}'.format(self.num_positives, self.num_cases)

        def pretty_print(self):
            result = '{} correct in {} .'.format(
                self.num_positives, self.num_cases)
            if self.num_cases != 0:
                result += ' Accuracy = {} . Avg # of choices = {} .'.format(
                    float(self.num_positives) / self.num_cases,
                    float(sum(self.num_choices)) / self.num_cases)
            else:
                result += ' Accuracy = 0 . Avg # of choices = 0 .'
            return result

    def __init__(self):
        self.accuracy_all = self.AccuracyStats()
        self.accuracy_subj = self.AccuracyStats()
        self.accuracy_obj = self.AccuracyStats()
        self.accuracy_pobj = self.AccuracyStats()

    def add_eval_result(self, label, correct, num_choices):
        self.accuracy_all.add_eval_result(correct, num_choices)
        if label == 'SUBJ':
            self.accuracy_subj.add_eval_result(correct, num_choices)
        elif label == 'OBJ':
            self.accuracy_obj.add_eval_result(correct, num_choices)
        elif label.startswith('PREP'):
            self.accuracy_pobj.add_eval_result(correct, num_choices)
        # self.check_consistency()

    def check_consistency(self):
        assert self.accuracy_all.num_cases == self.accuracy_subj.num_cases + \
            self.accuracy_obj.num_cases + self.accuracy_pobj.num_cases, \
            'Number of cases in 3 argument types do not sum to ' \
            'total number of cases.'
        assert self.accuracy_all.num_positives == \
            self.accuracy_subj.num_positives + \
            self.accuracy_obj.num_positives + \
            self.accuracy_pobj.num_positives, \
            'Number of positive cases in 3 argument types do not sum to ' \
            'total number of positive cases.'

    def add_eval_stats(self, eval_stats):
        # eval_stats.check_consistency()
        self.accuracy_all.add_accuracy_stats(eval_stats.accuracy_all)
        self.accuracy_subj.add_accuracy_stats(eval_stats.accuracy_subj)
        self.accuracy_obj.add_accuracy_stats(eval_stats.accuracy_obj)
        self.accuracy_pobj.add_accuracy_stats(eval_stats.accuracy_pobj)
        # self.check_consistency()

    def __str__(self):
        return 'All: {}\nSubj: {}\nObj: {}\nPObj: {}\n'.format(
            self.accuracy_all, self.accuracy_subj,
            self.accuracy_obj, self.accuracy_pobj)

    def pretty_print(self):
        # self.check_consistency()
        return 'All: {}\nSubj: {}\nObj: {}\nPObj: {}\n'.format(
            self.accuracy_all.pretty_print(), self.accuracy_subj.pretty_print(),
            self.accuracy_obj.pretty_print(), self.accuracy_pobj.pretty_print())


class EventScript:
    def __init__(self):
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
        for pred_token in sent.tokens:
            if pred_token.pos.startswith('VB'):
                # exclude "be" verbs
                if pred_token.lemma == 'be':
                    continue
                # TODO: exclude stop verbs
                # TODO: exclude verbs in quotes
                # exclude modifying verbs
                if sent.dep_graph.lookup_label(
                        'gov', pred_token.token_idx, 'xcomp'):
                    continue

                subj_list = sent.get_subj_list(pred_token.token_idx)
                obj_list = sent.get_obj_list(pred_token.token_idx)
                pobj_list = sent.get_pobj_list(pred_token.token_idx)

                if (not subj_list) and not (obj_list):
                    continue
                if not subj_list:
                    subj_list.append(None)
                if not obj_list:
                    obj_list.append(None)

                for arg_tuple in product(subj_list, obj_list):
                    self.add_event(Event.construct(
                        pred_token, # predicate
                        arg_tuple[0], # subject
                        arg_tuple[1], # object
                        pobj_list # prepositional object list
                    ))

    def read_from_document(self, doc):
        for sent in doc.sents:
            self.read_from_sentence(sent)
        for coref in doc.corefs:
            self.add_coref(coref)
        self.sort()

    def get_all_embeddings(self, model, syntax_suffix=False, head_only=False,
                           rep_only=False):
        for event in self.events:
            event.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
            event.get_eval_instances()

    def eval_most_freq_coref(self):
        eval_stats = EvalStats()
        coref_freqs = [len(coref.mentions) for coref in self.corefs]
        for event in self.events:
            for label, arg in event.get_all_arg():
                coref_freqs[arg.coref.idx] -= 1
                most_freq_idx = coref_freqs.index(max(coref_freqs))
                coref_freqs[arg.coref.idx] += 1
                eval_stats.add_eval_result(
                    label, (most_freq_idx == arg.coref.idx), len(self.corefs))
        return eval_stats

    def eval_most_sim_arg(self, model, use_other_args=False,
                          syntax_suffix=False, head_only=False, rep_only=False):
        self.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
        eval_stats = EvalStats()
        for event in self.events:
            for instance in event.eval_instances:
                eval_stats.add_eval_result(
                    instance.arg_label,
                    instance.eval_most_sim_arg(
                        model, self.corefs,
                        use_other_args, syntax_suffix, head_only, rep_only),
                    len(self.corefs))
        return eval_stats

    def eval_most_sim_event(self, model, use_max_score=False,
                            syntax_suffix=False, head_only=False,
                            rep_only=False):
        self.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
        eval_stats = EvalStats()
        event_embedding_list = [event.embedding for event in self.events]
        for idx in range(len(self.events)):
            for instance in self.events[idx].eval_instances:
                eval_stats.add_eval_result(
                    instance.arg_label,
                    instance.eval_most_sim_event(
                        model, self.corefs,
                        event_embedding_list[:idx] +
                        event_embedding_list[idx + 1:],
                        use_max_score, syntax_suffix, head_only, rep_only),
                    len(self.corefs))
        return eval_stats
