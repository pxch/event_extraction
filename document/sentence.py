from dependency import DependencyGraph
from token import Token
from util import get_class_name


class Sentence(object):
    def __init__(self, idx):
        # index of the sentence in the document
        self.idx = idx
        # list of all tokens in the sentence
        self.tokens = []
        # list of all dependencies in the sentence
        # excluding the root dependency
        self.deps = []
        # dependency graph built upon all dependencies
        self.dep_graph = None

    def add_token(self, token):
        assert isinstance(token, Token), \
            'add_token must be called with a {} instance'.format(
                get_class_name(Token))
        # set the sent_idx attrib of the token
        token.set_attrib('sent_idx', self.idx)
        # set the token_idx attrib of the token
        token.set_attrib('token_idx', len(self.tokens))
        self.tokens.append(token)

    def add_dep(self, dep):
        self.deps.append(dep)

    def get_token(self, idx):
        assert 0 <= idx < len(self.tokens), '{} out of token index'.format(idx)
        return self.tokens[idx]

    def __str__(self):
        return ' '.join([str(token) for token in self.tokens]) + '\t#DEP#\t' + \
               ' '.join([str(dep) for dep in self.deps])

    def pretty_print(self):
        return ' '.join([str(token) for token in self.tokens]) + '\n\t' + \
               ' '.join([str(dep) for dep in self.deps])

    def to_plain_text(self):
        return ' '.join([token.word for token in self.tokens])

    def build_dep_graph(self):
        self.dep_graph = DependencyGraph(self.idx, len(self.tokens))
        self.dep_graph.build(self.deps)

    def lookup_label(self, direction, token_idx, dep_label):
        return [self.get_token(idx) for idx in
                self.dep_graph.lookup_label(direction, token_idx, dep_label)]

    # add particles of a verb to its compound list
    def process_verb_prt(self):
        for pred_token in self.tokens:
            if pred_token.pos.startswith('VB'):
                if not pred_token.compounds_set:
                    pred_token.compounds.extend(self.lookup_label(
                        'gov', pred_token.token_idx, 'compound:prt'))
                    pred_token.compounds_set = True

    # get list of all subjective token indices for a predicate
    def get_subj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('gov', pred_idx, 'nsubj'))
        # agent of passive verb
        results.extend(self.lookup_label('gov', pred_idx, 'nmod:agent'))
        # controlling subject
        results.extend(self.lookup_label('gov', pred_idx, 'nsubj:xsubj'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all objective token indices for a predicate
    def get_obj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('gov', pred_idx, 'dobj'))
        # nsubjpass of passive verb
        results.extend(self.lookup_label('gov', pred_idx, 'nsubjpass'))
        # TODO: whether or not to include acl relation
        # results.extend(self.lookup_label('dep', pred_idx, 'acl'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all prepositional objective token indices for a predicate
    def get_pobj_list(self, pred_idx):
        results = []
        # look for all nmod dependencies
        for label, indices in self.dep_graph.lookup('gov', pred_idx).items():
            if label.startswith('nmod') and ':' in label:
                prep_label = label.split(':')[1]
                # exclude nmod:agent (subject)
                if prep_label != 'agent':
                    results.extend([(prep_label, self.get_token(idx))
                                    for idx in indices])
        return sorted(results, key=lambda pair: pair[1].token_idx)
