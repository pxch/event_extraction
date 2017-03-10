from collections import defaultdict
from warnings import warn
import consts


class Token:
    def __init__(self, word, lemma, pos):
        # word form of the token
        self.word = word.encode('ascii', 'ignore')
        # lemma form of the token
        self.lemma = lemma.encode('ascii', 'ignore')
        # part-of-speech tag of the token
        self.pos = pos
        # name entity tag of the token
        self.ner = ''
        # compounds
        self.compounds = []
        # boolean indicator to prevent adding duplicated compound words
        self.compounds_set = False
        # index of the sentence where the token is located
        # set in Sentence.add_token() method
        self.sent_idx = -1
        # index of the token in the sentence
        # set in Sentence.add_token() method
        self.token_idx = -1
        # pointer to the coreference chain which the token belongs to
        self.coref = None
        # pointer to the mention where the token belongs to
        self.mention = None

    def __str__(self):
        return '{}-{}/{}/{}'.format(
            self.token_idx, self.word, self.lemma, self.pos)

    def print_with_coref(self, include_compounds=False):
        result = '{}-{}'.format(self.token_idx, self.word)
        if include_compounds and self.compounds:
            result = '{}_{}'.format(
                result, '_'.join([token.word for token in self.compounds]))
        if self.coref is not None:
            result += '-entity#{:0>3d}'.format(self.coref.idx)
        return result

    # returns True if the token is a common noun or proper noun
    def is_noun(self):
        return self.pos.startswith('NN')

    # returns True if the token is a verb
    def is_verb(self):
        return self.pos.startswith('VB')

    # string form of the token
    def string_form(
            self, use_ner=False, use_lemma=False, include_compounds=False):
        result = ''
        # if use_ner is set, and the token has a valid ner tag, return the tag
        if use_ner and self.ner in consts.VALID_NER_TAGS:
            result = self.ner
        # if the token is a noun or a verb
        elif self.is_noun() or self.is_verb():
            # return the lower case of self.lemma if use_lemma is set
            if use_lemma:
                result = self.lemma.lower()
            # otherwise return the lower case of self.word
            else:
                result = self.word.lower()
            # if include_compounds is set, and the token has compound tokens
            # (currently only particles for verbs), concatenate the word/lemma
            # forms of all compound tokens to the output accordingly
            if include_compounds and self.compounds:
                for compound_token in self.compounds:
                    if use_lemma:
                        result += '_' + compound_token.lemma.lower()
                    else:
                        result += '_' + compound_token.word.lower()
        return result

    def set_attrib(self, attrib, value):
        assert attrib in self.__dict__, \
            '{} is not an attribute of class {}'.format(
                attrib, self.__class__.__name__)
        assert attrib not in ['word', 'lemma', 'pos'], \
            'Resetting the word/lemma/pos of a token after initialization ' \
            + 'is not allowed'
        self.__dict__[attrib] = value

    def add_coref_info(self, coref, mention):
        if self.mention is not None:
            warn('Token {} has existing mention {}'.format(self, self.mention))
            if self.mention.end_token_idx - self.mention.start_token_idx > \
                            mention.end_token_idx - mention.start_token_idx:
                warn(
                    'The new mention {} is nested in the exising mention, '
                    'ignore the new mention'.format(mention))
                return
            else:
                warn('Thew new mention {} has longer span than the existing '
                     'mention, override the existing mention'.format(mention))
        self.coref = coref
        self.mention = mention


class Dep:
    def __init__(self, label, gov_idx, dep_idx, extra=False):
        self.label = label.encode('ascii', 'ignore')
        self.gov_idx = gov_idx
        self.gov_token = None
        self.dep_idx = dep_idx
        self.dep_token = None
        self.extra = extra

    def __str__(self):
        return '{}-{}-{}'.format(self.label, self.gov_idx, self.dep_idx)

    def set_gov_token(self, token):
        self.gov_token = token

    def set_dep_token(self, token):
        self.dep_token = token


class DepGraph:
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
            'head token'.format(self.sent_idx, token_idx)
        return parent.items()[0][0], parent.items()[0][1][0]

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


class Sent:
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
        self.dep_graph = DepGraph(self.idx, len(self.tokens))
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
        return results

    # get list of all objective token indices for a predicate
    def get_obj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('gov', pred_idx, 'dobj'))
        # nsubjpass of passive verb
        results.extend(self.lookup_label('gov', pred_idx, 'nsubjpass'))
        # TODO: whether or not to include acl relation
        # results.extend(self.lookup_label('dep', pred_idx, 'acl'))
        return results

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
        return results
