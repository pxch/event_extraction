class Dep:
    def __init__(self, label, gov_idx, dep_idx, extra=False):
        self.label = label.encode('ascii', 'ignore')
        self.gov_idx = gov_idx
        self.dep_idx = dep_idx
        self.extra = extra

    def __str__(self):
        return '{}-{}-{}'.format(self.label, self.gov_idx, self.dep_idx)


class DepGraph:
    def __init__(self, sent_idx, num_tokens):
        self.sent_idx = sent_idx
        self.num_tokens = num_tokens
        self.governor_edges = [dict() for _ in range(num_tokens)]
        self.dependent_edges = [dict() for _ in range(num_tokens)]

    def build(self, deps):
        for dep in deps:
            if dep.label not in self.governor_edges[dep.gov_idx]:
                self.governor_edges[dep.gov_idx][dep.label] = [(dep.dep_idx, dep.extra)]
            else:
                self.governor_edges[dep.gov_idx][dep.label].append((dep.dep_idx, dep.extra))
            if dep.label not in self.dependent_edges[dep.dep_idx]:
                self.dependent_edges[dep.dep_idx][dep.label] = [(dep.gov_idx, dep.extra)]
            else:
                self.dependent_edges[dep.dep_idx][dep.label].append((dep.gov_idx, dep.extra))

    def lookup_label(self, direction, token_idx, dep_label, include_extra=True):
        assert direction in ['gov', 'dep'], 'input argument "direction" can only be gov/dep'
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
        assert direction in ['gov', 'dep'], 'input argument "direction" can only be gov/dep'
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

    def get_parent(self, token_idx):
        parent = self.lookup('dep', token_idx, include_extra=False)
        if len(parent) == 0:
            return 'root', -1
        assert len(parent) == 1 and len(parent.items()[0][1]) == 1, \
            'In Sentence #{}, Token #{} has more than 1 non-extra head token'.format(
                self.sent_idx, token_idx)
        return parent.items()[0][0], parent.items()[0][1][0]

    '''
    def lookup_gov(self, token_idx, dep_label, include_extra=True):
        assert 0 <= token_idx < self.num_tokens, \
            'Token idx {} out of range'.format(token_idx)
        results = self.governor_edges[token_idx].get(dep_label, [])
        if include_extra:
            results = [idx for idx, extra in results]
        else:
            results = [idx for idx, extra in results if not extra]
        return results

    def lookup_dep(self, token_idx, dep_label, include_extra=True):
        assert 0 <= token_idx < self.num_tokens, \
            'Token idx {} out of range'.format(token_idx)
        results = self.dependent_edges[token_idx].get(dep_label, [])
        if include_extra:
            results = [idx for idx, extra in results]
        else:
            results = [idx for idx, extra in results if not extra]
        return results
    '''

    def pretty_print(self):
        result = 'Num of tokens: {}\n'.format(self.num_tokens)
        result += 'Governor edges'
        for idx in range(self.num_tokens):
            result += '\n\tToken idx: {}\n'.format(idx)
            result += '\n'.join(['\t\t{}: {}'.format(label, edges)
                                 for label, edges in self.governor_edges[idx].items()])
        result += 'Dependent edges'
        for idx in range(self.num_tokens):
            result += '\n\tToken idx: {}\n'.format(idx)
            result += '\n'.join(['\t\t{}: {}'.format(label, edges)
                                 for label, edges in self.dependent_edges[idx].items()])
        return result
