from warnings import warn


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx):
        # index of the sentence where the mention is located
        self.sent_idx = sent_idx
        # index of the first token of the mention in the sentence
        self.start_token_idx = start_token_idx
        # index of the last token of the mention in the sentence
        self.end_token_idx = end_token_idx
        # index of the head token of the mention in the sentence
        # could be set in Document. method
        self.head_token_idx = -1
        # boolean indicator of whether the mention is a representative mention
        self.rep = False
        # text representation of the mention
        self.text = ''
        # index of the coreference chain which the mention belongs to
        # set in Coreference.add_mention() method
        self.coref_idx = -1
        # index of the mention in the coreference chain
        # set in Coreference.add_mention() method
        self.mention_idx = -1
        # list of all tokens in the mention
        # set in self.add_token_info() method
        self.tokens = []
        # pointer to the head token of the mention
        # set in self.add_token_info() method
        self.head_token = None
        # word form of the head token of the mention
        # set in self.add_token_info() method
        self.head_text = ''

    def __str__(self):
        return '{}-{}-{}-{}-{}-{}'.format(
            self.sent_idx, self.start_token_idx, self.end_token_idx,
            self.head_token_idx, self.rep, self.text)

    def set_attrib(self, attrib, value):
        assert attrib in self.__dict__, \
            '{} is not an attribute of class {}'.format(
                attrib, self.__class__.__name__)
        assert attrib not in ['sent_idx', 'start_token_idx', 'end_token_idx'], \
            'Resetting the sent_idx/start_token_idx/end_token_idx of ' \
            'a mention after initialization is not allowed'
        self.__dict__[attrib] = value

    def set_head_token_idx(self, dep_graph):
        if self.head_token_idx != -1:
            warn('Overwriting existing head_token_idx {}'.format(
                self.head_token_idx))
        self.head_token_idx = dep_graph.get_head_token_idx(
            self.start_token_idx, self.end_token_idx)

    def add_token_info(self, token_list):
        # set self.tokens
        if not self.tokens:
            self.tokens = token_list[self.start_token_idx:self.end_token_idx]
        # if self.text is not set, set it to the concatenation of the word
        # forms of all tokens
        if self.text == '':
            self.text = ' '.join([token.word for token in self.tokens])
        if self.head_token_idx != -1 and self.head_token is None:
            self.head_token = token_list[self.head_token_idx]
            self.head_text = self.head_token.word
