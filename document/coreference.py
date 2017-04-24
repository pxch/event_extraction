from warnings import warn


class Coreference(object):
    def __init__(self, idx):
        # index of the coreference chain in the document
        self.idx = idx
        # list of all mentions in the coreference chain
        self.mentions = []
        # pointer to the representative mention
        self.rep_mention = None

    def add_mention(self, mention):
        # set the coref_idx attrib of the mention
        mention.set_attrib('coref_idx', self.idx)
        # set the mention_idx attrib of the mention
        mention.set_attrib('mention_idx', len(self.mentions))
        self.mentions.append(mention)
        if mention.rep:
            self.rep_mention = mention

    def get_mention(self, idx):
        assert 0 <= idx < len(self.mentions), \
            '{} out of mention index'.format(idx)
        return self.mentions[idx]

    def __str__(self):
        return ' '.join([str(mention) for mention in self.mentions])

    def pretty_print(self):
        return 'entity#{:0>3d}    {}'.format(self.idx, self.rep_mention.text)

    def find_rep_mention(self):
        if self.rep_mention is not None:
            warn('Overriding existing rep_mention {}'.format(self.rep_mention))
        for mention in self.mentions:
            assert mention.head_token is not None, \
                'Cannot find the representative mention unless ' \
                'all mentions have head_token set'
            if mention.rep:
                warn('Overriding existing rep_mention {}'.format(mention))
                mention.rep = False
        # select mentions headed by proper nouns
        cand_indices = [mention.mention_idx for mention in self.mentions
                        if mention.head_token.pos.startswith('NNP')]
        # if no mentions are headed by proper nouns, select mentions headed
        # by common nouns
        if not cand_indices:
            cand_indices = [mention.mention_idx for mention in self.mentions
                            if mention.head_token.pos.startswith('NN')]
        # if no mentions are headed by either proper nouns or common noun,
        # use all mentions as candidates
        if not cand_indices:
            cand_indices = range(0, len(self.mentions))
        # select from candidate mentions the one with longest text
        cand_length = [len(self.get_mention(idx).text) for idx in cand_indices]
        rep_idx = cand_indices[cand_length.index(max(cand_length))]
        self.rep_mention = self.get_mention(rep_idx)
        self.rep_mention.set_attrib('rep', True)

