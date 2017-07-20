import pickle as pkl
from bz2 import BZ2File
from os.path import join

from corenlp import read_doc_from_corenlp
from node import SplitNode


class CoreNLPReader(object):
    def __init__(self, corenlp_dict):
        self.corenlp_dict = corenlp_dict

        self.doc = None
        self.idx_mapping = []
        self.corenlp_file_id = ''

    def read_corenlp_file(self, node):
        if node.file_id != self.corenlp_file_id:
            self.doc, self.idx_mapping = self.corenlp_dict[node.file_id]
            self.corenlp_file_id = node.file_id

    def parse_predicate(self, node):
        self.read_corenlp_file(node)
        mapped_idx = self.idx_mapping[node.sent_id].index(node.token_id)
        return self.doc.get_token(node.sent_id, mapped_idx).lemma.lower()

    def parse_node(self, node):
        if node.__class__ == SplitNode:
            # parse each node in the split node
            for n in node.node_list:
                self.parse_node(n)

            node.corenlp_idx_list = [idx for n in node.node_list
                                     for idx in n.corenlp_idx_list]
            node.corenlp_word_surface = \
                ' '.join([n.corenlp_word_surface for n in node.node_list])
            node.corenlp_lemma_surface = \
                ' '.join([n.corenlp_lemma_surface for n in node.node_list])

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
            self.read_corenlp_file(node)

            idx_mapping = self.idx_mapping[node.sent_id]
            node.corenlp_idx_list = [
                idx_mapping.index(idx) for idx in node.ptb_idx_list
                if idx in idx_mapping]

            if node.corenlp_idx_list:
                corenlp_sent = self.doc.get_sent(node.sent_id)
                node.corenlp_word_surface = ' '.join([
                    corenlp_sent.get_token(idx).word for idx
                    in node.corenlp_idx_list])
                node.corenlp_lemma_surface = ' '.join([
                    corenlp_sent.get_token(idx).lemma for idx
                    in node.corenlp_idx_list])

                node.head_idx = \
                    corenlp_sent.dep_graph.get_head_token_idx(
                        node.corenlp_idx_list[0],
                        node.corenlp_idx_list[-1] + 1)
                node.head_word = \
                    self.doc.get_token(
                        node.sent_id, node.head_idx).lemma.lower()

    @classmethod
    def build(cls, annotations, corenlp_root):
        print '\nBuilding CoreNLP Reader from {}'.format(corenlp_root)
        corenlp_dict = {}
        for predicate in annotations:
            node = predicate.node
            if node.file_id not in corenlp_dict:
                path = join(corenlp_root, 'parsed', node.directory,
                            node.file_id + '.xml.bz2')
                doc = read_doc_from_corenlp(BZ2File(path, 'r'))
                path = join(corenlp_root, 'idx', node.directory, node.file_id)
                idx_mapping = []
                with open(path, 'r') as fin:
                    for line in fin:
                        idx_mapping.append([int(i) for i in line.split()])
                corenlp_dict[node.file_id] = (doc, idx_mapping)
        return cls(corenlp_dict)

    @classmethod
    def load(cls, corenlp_dict_path):
        print '\nLoading CoreNLP Reader from {}\n'.format(corenlp_dict_path)
        corenlp_dict = pkl.load(open(corenlp_dict_path, 'r'))
        return cls(corenlp_dict)

    def save(self, corenlp_dict_path):
        print '\nSaving CoreNLP Redaer to {}\n'.format(corenlp_dict_path)
        pkl.dump(self.corenlp_dict, open(corenlp_dict_path, 'w'))
