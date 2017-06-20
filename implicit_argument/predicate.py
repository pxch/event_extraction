from node import Node, SplitNode
from operator import itemgetter
from collections import defaultdict


core_arg_list = ['arg0', 'arg1', 'arg2', 'arg3', 'arg4']


core_arg_mapping = {
    'bid': {'arg0': 'SUBJ', 'arg1': 'PREP_for', 'arg2': 'OBJ'},
    'sale': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_to',
             'arg3': 'PREP_for', 'arg4': 'PREP'},
    'loan': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_to', 'arg3': 'PREP',
             'arg4': 'PREP_at'},
    'cost': {'arg1': 'SUBJ', 'arg2': 'OBJ', 'arg3': 'PREP_to', 'arg4': 'PREP'},
    'plan': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_for',
             'arg3': 'PREP_for'},
    'investor': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_in'},
    'price': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_at', 'arg3': 'PREP'},
    'loss': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_to',
             'arg3': 'PREP_on'},
    'investment': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP_in'},
    'fund': {'arg0': 'SUBJ', 'arg1': 'OBJ', 'arg2': 'PREP', 'arg3': 'PREP'},
}


class Predicate(object):
    def __init__(self, node):
        self.node = node
        self.text = []
        self.imp_args = {}
        self.exp_args = {}
        self.n_pred = ''
        self.v_pred = ''
        self.candidates = []
        self.iarg0_dice = []
        self.iarg0_sim = []
        self.iarg1_dice = []
        self.iarg1_sim = []
        self.sum_dice = 0.0
        self.num_gt = 0.0
        self.num_model = 0.0
        self.thres = 0.0
        self.iarg0_candidate_pairs = []
        self.iarg1_candidate_pairs = []

    def __str__(self):
        result = '{}\t{}\n'.format(self.node, self.n_pred)
        if self.imp_args:
            for label in self.imp_args:
                result += 'implicit {}: {}\n'.format(
                    label,
                    ' '.join([str(node) for node in self.imp_args[label]]))
        if self.exp_args:
            for label in self.exp_args:
                result += '{}: {}\n'.format(
                    label,
                    ' '.join([str(node) for node in self.exp_args[label]]))
        return result

    def missing_arg(self, label):
        return label in core_arg_mapping[self.n_pred].keys() and \
               label not in self.exp_args

    def all_missing_args(self):
        return [label for label in core_arg_list if
                self.missing_arg(label)]

    def has_imp_arg(self, label, max_dist=-1):
        if label in self.imp_args:
            if max_dist == -1:
                return True
            else:
                for node in self.imp_args[label]:
                    if 0 <= self.node.sent_id - node.sent_id <= max_dist:
                        return True
        return False

    def num_imp_arg(self, max_dist=-1):
        return sum([1 for label in self.imp_args
                    if self.has_imp_arg(label, max_dist)])

    def has_oracle(self, label):
        candidates = map(itemgetter(0), self.candidates)
        if label in self.imp_args:
            for node in self.imp_args[label]:
                if node in candidates:
                    return True
                for cand in candidates:
                    if node.corenlp_surface == cand.corenlp_surface:
                        return True
                    if isinstance(node, Node) and isinstance(cand, Node) \
                            and node.sent_id == cand.sent_id \
                            and node.ptb_idx_list == cand.ptb_idx_list[1:] \
                            and node.subtree_pos[:-1] == cand.subtree_pos:
                        return True
        return False

    def num_oracle(self):
        return sum([1 for label in core_arg_mapping[self.n_pred].keys() if
                    self.has_oracle(label)])

    def add_args_from_ia(self, arg_list):
        tmp_imp_args = defaultdict(list)
        exp_args = defaultdict(list)
        for arg in arg_list:
            node = Node.parse(arg['node'])

            # ignore argument that is located in a sentence after the
            # predicate
            if node.file_id != self.node.file_id or node.sent_id > \
                    self.node.sent_id:
                continue

            label = arg['label'].lower()
            attribute = arg['attribute']

            # add explicit arguments to exp_args
            if attribute == 'Explicit':
                exp_args[label].append(node)
                # remove the label from tmp_imp_args, as we do not process
                # an implicit argument if some explicit arguments with
                # the same label exist
                tmp_imp_args.pop(label, None)

            # add non-explicit arguments to tmp_imp_args
            else:
                # do not add the argument when some explicit arguments with
                # the same label exist
                if label not in exp_args:
                    tmp_imp_args[label].append((node, attribute))

        # remove incorporated arguments from tmp_imp_args
        # incorporated argument: argument with the same node as
        # the predicate itself
        for label, fillers in tmp_imp_args.items():
            if self.node in [node for node, attribute in fillers]:
                tmp_imp_args.pop(label, None)

        # process split arguments
        imp_args = {}
        for label, fillers in tmp_imp_args.items():
            # add non-split arguments to imp_args
            imp_args[label] = [node for node, attribute in fillers
                               if attribute == '']
            split_nodes = [node for node, attribute in fillers
                           if attribute == 'Split']
            sent_ids = set([node.sent_id for node in split_nodes])

            # group split arguments by their sentence id,
            # and sort components by token id within each group
            grouped_split_nodes = []
            for sent_id in sent_ids:
                grouped_split_nodes.append(sorted(
                    [node for node in split_nodes if node.sent_id == sent_id],
                    key=lambda n: n.token_id))

            # add each split node to imp_args
            for ss in grouped_split_nodes:
                assert len(ss) > 1, \
                    'SplitNode {0} contains only one node'.format(ss)
                imp_args[label].append(SplitNode(ss))

        self.imp_args = imp_args
        self.exp_args = exp_args

    def check_sent_dist(self, sent_id):
        return 0 <= self.node.sent_id - sent_id <= 2

    def add_candidates(self, instances):
        for instance in instances:
            if self.check_sent_dist(instance.sent_id):
                pred_node = None
                for label, node in instance.arguments:
                    if label.startswith('rel'):
                        pred_node = node
                        break
                if pred_node is not None:
                    for label, node in instance.arguments:
                        if label[:4] in core_arg_list:
                            if node not in map(itemgetter(0), self.candidates):
                                self.candidates.append((node, pred_node))
                else:
                    print 'Error! Cannot find rel argument from ' \
                          'PropBank/NomBank: {}'.format(instance)

    def filterExpArgsFromCandidates(self):
        filtered_candidates = []
        for label, nodes in self.exp_args.items():
            if label in core_arg_list:
                matched_candidates = []
                split_nodes = []
                for node in nodes:
                    # keep single nodes in explicit arguments that are in
                    # Nombank
                    if node in map(itemgetter(0), self.candidates):
                        matched_candidates.append(node)
                    else:
                        split_nodes.append(node)
                # for single nodes that are not in Nombank, combine them to
                # match split nodes in Nombank
                while split_nodes:
                    find_match = False
                    for candidate in map(itemgetter(0), self.candidates):
                        if candidate.__class__ == SplitNode:
                            if split_nodes[0] in candidate.pieces:
                                matched_candidates.append(candidate)
                                split_nodes = list(
                                    set(split_nodes) - set(candidate.pieces))
                                find_match = True
                                break
                    if not find_match:
                        raise AssertionError(
                            '{0} not in candidates {1}'.format(
                                split_nodes[0],
                                map(str, map(itemgetter(0), self.candidates))))
                self.exp_args[label] = matched_candidates
                filtered_candidates.extend(matched_candidates)
        # remove matched explicit arguments from candidates
        self.candidates = [candidate for candidate in self.candidates if
                           candidate[0] not in filtered_candidates]

    def checkExpArgs(self):
        for label, nodes in self.exp_args.items():
            if label in core_arg_list:
                for node in nodes:
                    assert node not in map(itemgetter(0), self.candidates), \
                        'Explicit argument {0} still exists in candidates {1}' \
                        ' after filtering' \
                            .format(node, map(itemgetter(0), self.candidates))

    def eval(self, thres):
        assert (self.mis_arg0 and self.iarg0_sim) or \
               (not self.mis_arg0 and not self.iarg0_sim), \
            'Cosine similarity for candidates on iarg0 slot should only exist ' \
            'when arg0 is missing.'
        assert (self.mis_arg1 and self.iarg1_sim) or \
               (not self.mis_arg1 and not self.iarg1_sim), \
            'Cosine similarity for candidates on iarg1 slot should only exist ' \
            'when arg1 is missing.'
        assert (self.imp_arg0 or all(dice == 0.0 for dice in
                                     self.iarg0_dice)), \
            'Dice score for candidates on iarg0 slot should all be 0 if no ' \
            'iarg0 is labeled.'
        assert (self.imp_arg1 or all(dice == 0.0 for dice in self.iarg1_dice)), \
            'Dice score for candidates on iarg1 slot should all be 0 if no iarg1 is labeled.'

        self.sum_dice = 0.0

        self.num_gt = 0
        if self.imp_arg0:
            self.num_gt += 1
        if self.imp_arg1:
            self.num_gt += 1

        self.num_model = 0
        if self.mis_arg0 and self.mis_arg1:
            assert len(self.iarg0_sim) == len(
                self.iarg1_sim), 'Number of iarg0 and iarg1 fillers mismatch'
            for idx in range(len(self.iarg0_sim)):
                if self.iarg0_sim[idx] >= self.iarg1_sim[idx]:
                    self.iarg1_sim[idx] = 0.0
                else:
                    self.iarg0_sim[idx] = 0.0

        if self.mis_arg0:
            max_iarg0_sim = max(self.iarg0_sim)
            if max_iarg0_sim >= thres:
                self.num_model += 1
                max_iarg0_sim_idx = [idx for idx, val in
                                     enumerate(self.iarg0_sim) if
                                     val == max_iarg0_sim]
                max_iarg0_dice = max(
                    [self.iarg0_dice[idx] for idx in max_iarg0_sim_idx])
                self.sum_dice += max_iarg0_dice

        if self.mis_arg1:
            max_iarg1_sim = max(self.iarg1_sim)
            if max_iarg1_sim >= thres:
                self.num_model += 1
                max_iarg1_sim_idx = [idx for idx, val in
                                     enumerate(self.iarg1_sim) if
                                     val == max_iarg1_sim]
                max_iarg1_dice = max(
                    [self.iarg1_dice[idx] for idx in max_iarg1_sim_idx])
                self.sum_dice += max_iarg1_dice
