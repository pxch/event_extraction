class Mention:
    def __init__(self, sent_idx, start_token_idx, end_token_idx,
                 head_token_idx, rep, text):
        self.sent_idx = sent_idx
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.head_token_idx = head_token_idx
        self.rep = rep
        self.text = text.encode('ascii', 'ignore')
        self.head_text = ''
        self.tokens = []
        self.head_token = None

    def __str__(self):
        return '{}-{}-{}-{}-{}-{}'.format(self.sent_idx, self.start_token_idx,
                                          self.end_token_idx, self.head_token_idx, self.rep, self.text)

    def add_token_info(self, token_list):
        for idx in range(self.start_token_idx, self.end_token_idx):
            self.tokens.append(token_list[idx])
        self.head_token = token_list[self.head_token_idx]
        self.head_text = self.head_token.word

    def get_embedding(self, model, suffix, head_only=False):
        embedding = model.zeros()
        if head_only:
            head_embedding = self.head_token.get_embedding(model, suffix)
            if head_embedding is not None:
                embedding += head_embedding
        else:
            for token in self.tokens:
                token_embedding = token.get_embedding(model, suffix)
                if token_embedding is not None:
                    embedding += token_embedding
        return embedding


class Coref:
    def __init__(self, idx):
        self.idx = idx
        self.mentions = []
        self.rep_mention = None
        self.occ_count = 0

    def add_mention(self, mention):
        self.mentions.append(mention)
        if mention.rep:
            self.rep_mention = mention

    def __str__(self):
        return ' '.join([str(mention) for mention in self.mentions])

    def pretty_print_rep(self):
        return 'entity#{:0>3d}    {}'.format(self.idx, self.rep_mention.text)

    def get_embedding(self, model, suffix, head_only=False, rep_only=True):
        embedding = model.zeros()
        if rep_only:
            embedding += self.rep_mention.get_embedding(model, suffix, head_only)
        else:
            for mention in self.mentions:
                embedding += mention.get_embedding(model, suffix, head_only)
        return embedding
