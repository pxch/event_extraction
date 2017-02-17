
class Dep:
  def __init__(self, label, gov_idx, dep_idx):
    self.label = label.encode('ascii', 'ignore')
    # TODO: add boolean value "extra" from the new nndep parser
    self.gov_idx = gov_idx
    self.dep_idx = dep_idx

  def __str__(self):
    return '{}-{}-{}'.format(self.label, self.gov_idx, self.dep_idx)

class DepGraph:
  def __init__(self, num_tokens):
    self.num_tokens = num_tokens
    self.governor_edges = [dict() for idx in range(num_tokens)]
    self.dependent_edges = [dict() for idx in range(num_tokens)]

  def build(self, deps):
    for dep in deps:
      if dep.label not in self.governor_edges[dep.gov_idx]:
        self.governor_edges[dep.gov_idx][dep.label] = [dep.dep_idx]
      else:
        self.governor_edges[dep.gov_idx][dep.label].append(dep.dep_idx)
      if dep.label not in self.dependent_edges[dep.dep_idx]:
        self.dependent_edges[dep.dep_idx][dep.label] = [dep.gov_idx]
      else:
        self.dependent_edges[dep.dep_idx][dep.label].append(dep.gov_idx)

  def lookup_gov(self, token_idx, dep_label):
    assert token_idx >= 0 and token_idx < self.num_tokens, \
        'Token idx {} out of range'.format(token_idx)
    return self.governor_edges[token_idx].get(dep_label, [])

  def lookup_dep(self, token_idx, dep_label):
    assert token_idx >= 0 and token_idx < self.num_tokens, \
        'Token idx {} out of range'.format(token_idx)
    return self.dependent_edges[token_idx].get(dep_label, [])

  def pretty_print(self):
    result = 'Num of tokens: {}\n'.format(self.num_tokens)
    result += 'Governor edges'
    for idx in range(self.num_tokens):
      result += '\n\tToken idx: {}\n'.format(idx)
      result += '\n'.join(['\t\t{}: {}'.format(label, edges) \
          for label, edges in self.governor_edges[idx].items()])
    result += 'Dependent edges'
    for idx in range(self.num_tokens):
      result += '\n\tToken idx: {}\n'.format(idx)
      result += '\n'.join(['\t\t{}: {}'.format(label, edges) \
          for label, edges in self.dependent_edges[idx].items()])
    return result

