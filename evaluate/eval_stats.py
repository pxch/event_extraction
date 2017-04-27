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
                result += ' Accuracy = {:.2f}% .'.format(
                    float(self.num_positives) * 100. / self.num_cases)
                result += ' Avg # of choices = {:.2f} .'.format(
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
