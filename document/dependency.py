from collections import defaultdict
from warnings import warn


class Dependency(object):
    def __init__(self, label, gov_idx, dep_idx, extra=False):
        self.label = label.encode('ascii', 'ignore')
        self.gov_idx = gov_idx
        self.gov_token = None
        self.dep_idx = dep_idx
        self.dep_token = None
        self.extra = extra
        if self.label == 'nsubjpass:xsubj':
            self.extra = True

    def __str__(self):
        return '{}-{}-{}'.format(self.label, self.gov_idx, self.dep_idx)

    def set_gov_token(self, token):
        self.gov_token = token

    def set_dep_token(self, token):
        self.dep_token = token


class DependencyGraph(object):
    def __init__(self, sent_idx, num_tokens):
        self.sent_idx = sent_idx
        self.num_tokens = num_tokens
        self.governor_edges = [defaultdict(list) for _ in range(num_tokens)]
        self.dependent_edges = [defaultdict(list) for _ in range(num_tokens)]

    def build(self, deps):
        for dep in deps:
            self.governor_edges[dep.gov_idx][dep.label].append(
                (dep.dep_idx, dep.extra))
            self.dependent_edges[dep.dep_idx][dep.label].append(
                (dep.gov_idx, dep.extra))

    def __str__(self):
        result = 'Num of tokens: {}\n'.format(self.num_tokens)
        result += 'Governor edges'
        for idx in range(self.num_tokens):
            result += '\n\tToken idx: {}\n'.format(idx)
            result += '\n'.join(['\t\t{}: {}'.format(label, edges)
                                 for label, edges in
                                 self.governor_edges[idx].items()])
        result += 'Dependent edges'
        for idx in range(self.num_tokens):
            result += '\n\tToken idx: {}\n'.format(idx)
            result += '\n'.join(['\t\t{}: {}'.format(label, edges)
                                 for label, edges in
                                 self.dependent_edges[idx].items()])
        return result

    def lookup_label(self, direction, token_idx, dep_label, include_extra=True):
        assert direction in ['gov', 'dep'], \
            'input argument "direction" can only be gov/dep'
        assert 0 <= token_idx < self.num_tokens, \
            'Token idx {} out of range'.format(token_idx)
        if direction == 'gov':
            edges = self.governor_edges[token_idx].get(dep_label, [])
        else:
            edges = self.dependent_edges[token_idx].get(dep_label, [])
        if include_extra:
            results = [idx for idx, extra in edges]
        else:
            results = [idx for idx, extra in edges if not extra]
        return results

    def lookup(self, direction, token_idx, include_extra=True):
        assert direction in ['gov', 'dep'], \
            'input argument "direction" can only be gov/dep'
        assert 0 <= token_idx < self.num_tokens, \
            'Token idx {} out of range'.format(token_idx)
        if direction == 'gov':
            all_edges = self.governor_edges[token_idx]
        else:
            all_edges = self.dependent_edges[token_idx]
        results = {}
        for label, edges in all_edges.items():
            if include_extra:
                indices = [idx for idx, extra in edges]
            else:
                indices = [idx for idx, extra in edges if not extra]
            if indices:
                results[label] = indices
        return results

    # get the parent token index of the input token index
    # return -1 if the input token is root
    def get_parent(self, token_idx):
        parent = self.lookup('dep', token_idx, include_extra=False)
        if len(parent) == 0:
            return 'root', -1
        assert len(parent) == 1 and len(parent.items()[0][1]) == 1, \
            'In Sentence #{}, Token #{} has more than 1 non-extra ' \
            'head token: {}'.format(self.sent_idx, token_idx, parent)
        return parent.items()[0][0], parent.items()[0][1][0]

    def get_root_path_length(self, token_idx):
        path_length = 0
        current_idx = token_idx
        while current_idx != -1:
            _, current_idx = self.get_parent(current_idx)
            path_length += 1
        return path_length

    # get the head token index from a range of tokens
    def get_head_token_idx(self, start_token_idx, end_token_idx):
        assert 0 <= start_token_idx < self.num_tokens, \
            'Start token idx {} out of range'.format(start_token_idx)
        assert 0 < end_token_idx <= self.num_tokens, \
            'End token idx {} out of range'.format(end_token_idx)
        assert start_token_idx < end_token_idx, \
            'Start token idx {} not smaller than end token idx {}'.format(
                start_token_idx, end_token_idx)
        head_idx_map = []
        for token_idx in range(start_token_idx, end_token_idx):
            head_trace = [token_idx]
            while start_token_idx <= head_trace[-1] < end_token_idx:
                _, head_idx = self.get_parent(head_trace[-1])
                # warn if there is a loop in finding one token's head token
                if head_idx in head_trace:
                    warn('In sentence #{}, token #{} has loop in its '
                         'head trace list.'.format(self.sent_idx, token_idx))
                    break
                head_trace.append(head_idx)
            head_idx_map.append((token_idx, head_trace[-2]))
        head_idx_list = [head_idx for _, head_idx in head_idx_map]
        # warn if the tokens in the range don't have the same head token
        if min(head_idx_list) != max(head_idx_list):
            warn('In sentence #{}, tokens within the range [{}, {}] do not '
                 'have the same head token'.format(self.sent_idx,
                                                   start_token_idx,
                                                   end_token_idx))
        return min(head_idx_list)
