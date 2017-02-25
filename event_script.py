class EvalStats:
    def __init__(self):
        self.num_all = 0
        self.num_all_correct = 0
        self.num_all_args = 0
        self.num_subj = 0
        self.num_subj_correct = 0
        self.num_subj_args = 0
        self.num_obj = 0
        self.num_obj_correct = 0
        self.num_obj_args = 0
        self.num_pobj = 0
        self.num_pobj_correct = 0
        self.num_pobj_args = 0

    def add_eval_result(self, label, correct, num_args):
        self.num_all += 1
        self.num_all_correct += correct
        self.num_all_args += num_args
        if label == 'SUBJ':
            self.num_subj += 1
            self.num_subj_correct += correct
            self.num_subj_args += num_args
        elif label == 'OBJ':
            self.num_obj += 1
            self.num_obj_correct += correct
            self.num_obj_args += num_args
        elif label.startswith('PREP'):
            self.num_pobj += 1
            self.num_pobj_correct += correct
            self.num_pobj_args += num_args

    def check_consistency(self):
        assert self.num_all == self.num_subj + self.num_obj + self.num_pobj and \
               self.num_all_correct == self.num_subj_correct + \
                                       self.num_obj_correct + self.num_pobj_correct and \
               self.num_all_args == self.num_subj_args + \
                                    self.num_obj_args + self.num_pobj_args, \
            'Numbers of 3 argument types do not sum to number of all arguments.'

    def add_eval_stats(self, eval_stats):
        eval_stats.check_consistency()
        self.num_all += eval_stats.num_all
        self.num_all_correct += eval_stats.num_all_correct
        self.num_all_args += eval_stats.num_all_args
        self.num_subj += eval_stats.num_subj
        self.num_subj_correct += eval_stats.num_subj_correct
        self.num_subj_args += eval_stats.num_subj_args
        self.num_obj += eval_stats.num_obj
        self.num_obj_correct += eval_stats.num_obj_correct
        self.num_obj_args += eval_stats.num_obj_args
        self.num_pobj += eval_stats.num_pobj
        self.num_pobj_correct += eval_stats.num_pobj_correct
        self.num_pobj_args += eval_stats.num_pobj_args
        self.check_consistency()

    @staticmethod
    def print_accuracy(prefix, num_all, num_correct, num_args):
        print '\t{}: {} correct in {}. Accuracy = {}. Avg # of choices = {}'.format(
            prefix, num_correct, num_all,
            float(num_correct) / num_all if num_all != 0 else 0,
            float(num_args) / num_all if num_all != 0 else 0)

    def pretty_print(self):
        self.check_consistency()
        self.print_accuracy('All', self.num_all, self.num_all_correct, self.num_all_args)
        self.print_accuracy('Subj', self.num_subj, self.num_subj_correct, self.num_subj_args)
        self.print_accuracy('Obj', self.num_obj, self.num_obj_correct, self.num_obj_args)
        self.print_accuracy('PObj', self.num_pobj, self.num_pobj_correct, self.num_pobj_args)


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
        results += '\n'.join([coref.pretty_print_rep() for coref in self.corefs])
        return results

    def get_all_arg_coref_idx(self):
        return [coref_idx for event in self.events
                for coref_idx in event.get_all_arg_coref_idx()]

    def get_all_embeddings(self, model, syntax_suffix=False, head_only=False, rep_only=False):
        for event in self.events:
            event.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
            event.get_eval_instances()

    def eval_most_freq_coref(self):
        eval_stats = EvalStats()
        coref_freqs = [len(coref.mentions) for coref in self.corefs]
        for event in self.events:
            for label, arg in event.get_all_arg():
                coref_freqs[arg.coref_idx] -= 1
                most_freq_idx = coref_freqs.index(max(coref_freqs))
                coref_freqs[arg.coref_idx] += 1
                eval_stats.add_eval_result(label, (most_freq_idx == arg.coref_idx), len(self.corefs))
        return eval_stats
        '''
        num_all = 0
        num_correct = 0
        coref_freqs = [len(coref.mentions) for coref in self.corefs]
        for event in self.events:
          for arg in event.get_all_arg():
            num_all += 1
            coref_freqs[arg.coref_idx] -= 1
            most_freq_idx = coref_freqs.index(max(coref_freqs))
            coref_freqs[arg.coref_idx] += 1
            if most_freq_idx == arg.coref_idx:
              num_correct += 1
        return num_all, num_correct
        '''

    def eval_most_sim_arg(self, model, use_other_args=False,
                          syntax_suffix=False, head_only=False, rep_only=False):
        eval_stats = EvalStats()
        for event in self.events:
            for instance in event.eval_instances:
                eval_stats.add_eval_result(
                    instance.arg_label,
                    instance.eval_most_sim_arg(model, self.corefs,
                                               use_other_args, syntax_suffix, head_only, rep_only),
                    len(self.corefs))
        return eval_stats
        '''
        num_all = 0
        num_correct = 0
        for event in self.events:
          for instance in event.eval_instances:
            num_all += 1
            if instance.eval_most_sim_arg(model, self.corefs, \
                use_other_args, syntax_suffix, head_only, rep_only):
              num_correct += 1
        return num_all, num_correct
        '''

    def eval_most_sim_event(self, model, use_max_score=False,
                            syntax_suffix=False, head_only=False, rep_only=False):
        eval_stats = EvalStats()
        event_embedding_list = [event.embedding for event in self.events]
        for idx in range(len(self.events)):
            for instance in self.events[idx].eval_instances:
                eval_stats.add_eval_result(
                    instance.arg_label,
                    instance.eval_most_sim_event(model, self.corefs,
                                                 event_embedding_list[:idx] + event_embedding_list[idx + 1:],
                                                 use_max_score, syntax_suffix, head_only, rep_only),
                    len(self.corefs))
        return eval_stats
        '''
        num_all = 0
        num_correct = 0
        event_embedding_list = [event.embedding for event in self.events]
        for idx in range(len(self.events)):
          for instance in self.events[idx].eval_instances:
            num_all += 1
            if instance.eval_most_sim_event(model, self.corefs, \
                event_embedding_list[:idx] + event_embedding_list[idx+1:], \
                use_max_score, syntax_suffix, head_only, rep_only):
              num_correct += 1
        return num_all, num_correct
        '''
