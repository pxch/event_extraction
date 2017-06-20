import os
from bz2 import BZ2File

from nltk.corpus import BracketParseCorpusReader
from nltk.stem import WordNetLemmatizer

from corenlp import read_doc_from_corenlp
from node import Node, SplitNode


class PTBReader(object):
    def __init__(self, corpus_root, file_pattern, corenlp_dict):
        self.corpus_root = corpus_root
        self.file_pattern = file_pattern
        self.corenlp_dict = corenlp_dict

        self.ptb = BracketParseCorpusReader(corpus_root, file_pattern)
        self.lemmatizer = WordNetLemmatizer()

        self.all_sents = []
        self.all_tagged_sents = []
        self.all_parsed_sents = []
        self.ptb_file_id = ''

        self.doc = None
        self.idx_mapping = []
        self.corenlp_file_id = ''

    def read_ptb_file(self, node):
        if node.file_id != self.ptb_file_id:
            path = '{0}/{1}.mrg'.format(node.directory, node.file_id)
            self.all_sents = self.ptb.sents(fileids=path)
            self.all_tagged_sents = self.ptb.tagged_sents(fileids=path)
            self.all_parsed_sents = self.ptb.parsed_sents(fileids=path)
            self.ptb_file_id = node.file_id

    def read_corenlp_file(self, node):
        if node.file_id != self.corenlp_file_id:
            self.doc, self.idx_mapping = self.corenlp_dict[node.file_id]
            self.corenlp_file_id = node.file_id

    def get_subtree_pos(self, node):
        parsed_sent = self.all_parsed_sents[node.sent_id]
        token_pos = parsed_sent.leaf_treeposition(node.token_id)
        subtree_pos = token_pos[:-(node.phrase_level + 1)]
        return subtree_pos

    def is_child_node(self, parent, child):
        if not (isinstance(parent, Node) and isinstance(child, Node)):
            return False
        if not (parent.file_id == child.file_id
                and parent.sent_id == child.sent_id):
            return False

        self.read_ptb_file(parent)
        parent_subtree_pos = self.get_subtree_pos(parent)
        child_subtree_pos = self.get_subtree_pos(child)
        if child_subtree_pos[:len(parent_subtree_pos)] == parent_subtree_pos:
            return True
        else:
            return False

    def parse_single_node(self, node):
        self.read_ptb_file(node)
        self.read_corenlp_file(node)

        node.ptb_idx_list.append(node.token_id)
        node.ptb_surface = self.all_tagged_sents[node.sent_id][node.token_id][0]

        mapped_idx = self.idx_mapping[node.sent_id].index(node.token_id)

        node.corenlp_idx_list.append(mapped_idx)
        node.corenlp_surface = \
            self.doc.get_token(node.sent_id, mapped_idx).word

        node.head_idx = mapped_idx
        node.head_word = \
            self.doc.get_token(node.sent_id, mapped_idx).lemma.lower()

        return node.head_word

    def parse_node(self, node):
        if node.__class__ == SplitNode:
            # parse each node in the split node
            for n in node.node_list:
                self.parse_node(n)

            # combine the ptb_surface of each node
            node.ptb_idx_list = [idx for n in node.node_list
                                 for idx in n.ptb_idx_list]
            node.ptb_surface = ' '.join([n.ptb_surface for n in node.node_list])

            node.corenlp_idx_list = [idx for n in node.node_list
                                     for idx in n.corenlp_idx_list]
            node.corenlp_surface = \
                ' '.join([n.corenlp_surface for n in node.node_list])

            node.head_node = None
            min_root_path_length = 999

            corenlp_sent = self.doc.get_sent(node.sent_id)

            for n in node.node_list:
                try:
                    root_path_length = \
                        corenlp_sent.dep_graph.get_root_path_length(n.head_idx)
                    # with same root_path_length, take the latter token
                    if root_path_length <= min_root_path_length:
                        min_root_path_length = root_path_length
                        node.head_node = n
                except AssertionError as detail:
                    print detail
                    print corenlp_sent
                    print corenlp_sent.dep_graph
                    print n, n.ptb_surface, n.corenlp_idx_list, n.head_idx

            node.head_idx = node.head_node.head_idx
            node.head_word = node.head_node.head_word

        else:
            self.read_ptb_file(node)

            node.subtree_pos = self.get_subtree_pos(node)

            parsed_sent = self.all_parsed_sents[node.sent_id]
            node.ptb_idx_list = []
            for idx in range(len(parsed_sent.leaves())):
                if parsed_sent.leaf_treeposition(idx)[:len(node.subtree_pos)] \
                        == node.subtree_pos:
                    node.ptb_idx_list.append(idx)

            assert node.ptb_idx_list == \
                range(node.ptb_idx_list[0], node.ptb_idx_list[-1] + 1), \
                'Error in matching indices for subtree leaves: {0}'.format(node)

            tagged_sent = self.all_tagged_sents[node.sent_id]
            node.ptb_surface = ' '.join(
                [word[0] for word in [
                    tagged_sent[i] for i in node.ptb_idx_list]])

            self.read_corenlp_file(node)

            idx_mapping = self.idx_mapping[node.sent_id]
            node.corenlp_idx_list = [
                idx_mapping.index(idx) for idx in node.ptb_idx_list
                if idx in idx_mapping]

            if node.corenlp_idx_list:
                corenlp_sent = self.doc.get_sent(node.sent_id)
                node.corenlp_surface = ' '.join(
                    [corenlp_sent.get_token(idx).word
                     for idx in node.corenlp_idx_list])
                node.head_idx = \
                    corenlp_sent.dep_graph.get_head_token_idx(
                        node.corenlp_idx_list[0],
                        node.corenlp_idx_list[-1] + 1)
                node.head_word = \
                    self.doc.get_token(
                        node.sent_id, node.head_idx).lemma.lower()
