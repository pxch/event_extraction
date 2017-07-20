from util import get_class_name


class Node(object):
    def __init__(self, file_id, sent_id, token_id, phrase_level):
        self.directory = file_id.split('_')[1][:2]
        self.file_id = file_id
        self.sent_id = int(sent_id)
        self.token_id = int(token_id)
        self.phrase_level = int(phrase_level)

        # Penn TreeBank related info
        self.ptb_idx_list = []
        self.ptb_surface = ''
        self.subtree_pos = []

        # Stanford CoreNLP related info
        self.corenlp_idx_list = []
        self.corenlp_word_surface = ''
        self.corenlp_lemma_surface = ''

        self.head_idx = -1
        self.head_word = ''

    @staticmethod
    def parse(text):
        items = text.split(',')
        if len(items) > 1:
            return SplitNode([Node.parse(item) for item in items])
        return Node.from_text(text)

    @classmethod
    def from_text(cls, text):
        items = text.split(':')
        if len(items) != 4:
            raise ParseNodeError(
                'expected 4 parts, separated by :, got {}: {}'.format(
                    len(items), text))
        return cls(items[0], items[1], items[2], items[3])

    def to_text(self):
        return '{}:{}:{}:{}'.format(
            self.file_id, self.sent_id, self.token_id, self.phrase_level)

    def __str__(self):
        return self.to_text()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(
            (self.file_id, self.sent_id, self.token_id, self.phrase_level))


class SplitNode(object):
    def __init__(self, node_list):
        if len(node_list) <= 1:
            raise ParseNodeError(
                'only 1 node provided in {}'.format(map(str, node_list)))
        if not all(isinstance(n, Node) for n in node_list):
            raise ParseNodeError(
                'every node must be a {} instance'.format(get_class_name(Node)))
        if not all(node_list[0].file_id == n.file_id for n in node_list[1:]):
            raise ParseNodeError(
                'inconsistency in file_id in {}'.format(map(str, node_list)))
        if not all(node_list[0].sent_id == n.sent_id for n in node_list[1:]):
            raise ParseNodeError(
                'inconsistency in sent_id in {}'.format(map(str, node_list)))
        self.node_list = sorted(node_list, key=lambda n: n.token_id)
        self.file_id = node_list[0].file_id
        self.sent_id = node_list[0].sent_id

        # Penn TreeBank related info
        self.ptb_idx_list = []
        self.ptb_surface = ''

        # Stanford CoreNLP related info
        self.corenlp_idx_list = []
        self.corenlp_word_surface = ''
        self.corenlp_lemma_surface = ''

        self.head_idx = -1
        self.head_word = ''

        self.head_node = None

    @classmethod
    def from_text(cls, text):
        items = text.split(',')
        if len(items) <= 1:
            raise ParseNodeError(
                'SplitNode.from_text expects input text with at least '
                '2 parts, separated by ",", got {}: {}'.format(
                    len(items), text))
        return cls([Node.parse(item) for item in items])

    def to_text(self):
        return ','.join(map(str, self.node_list))

    def __str__(self):
        return ','.join(map(str, self.node_list))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self.node_list))


class ParseNodeError(Exception):
    pass
