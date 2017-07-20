from nltk.corpus import BracketParseCorpusReader

from node import Node, SplitNode


class PTBReader(object):
    def __init__(self, corpus_root, file_pattern):
        self.ptb = BracketParseCorpusReader(corpus_root, file_pattern)

        self.all_sents = []
        self.all_tagged_sents = []
        self.all_parsed_sents = []
        self.ptb_file_id = ''

    def read_ptb_file(self, node):
        if node.file_id != self.ptb_file_id:
            path = '{0}/{1}.mrg'.format(node.directory, node.file_id)
            self.all_sents = self.ptb.sents(fileids=path)
            self.all_tagged_sents = self.ptb.tagged_sents(fileids=path)
            self.all_parsed_sents = self.ptb.parsed_sents(fileids=path)
            self.ptb_file_id = node.file_id

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

    def parse_node(self, node):
        if node.__class__ == SplitNode:
            # parse each node in the split node
            for n in node.node_list:
                self.parse_node(n)

            # combine the ptb_surface of each node
            node.ptb_idx_list = [idx for n in node.node_list
                                 for idx in n.ptb_idx_list]
            node.ptb_surface = ' '.join([n.ptb_surface for n in node.node_list])

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
