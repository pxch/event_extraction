from operator import attrgetter

valid_pos = ['NN', 'VB']
valid_ner = ['PERSON', 'LOCATION', 'ORGANIZATION']

class Token:
  def __init__(self, word, lemma, pos, ner):
    # basic info
    self.word = word.encode('ascii', 'ignore')
    self.lemma = lemma.encode('ascii', 'ignore')
    self.pos = pos
    self.ner = ner
    # compounds
    self.compounds = []
    # index info
    self.sent_idx = -1
    self.token_idx = -1
    # coref info
    self.coref_idx = -1
    self.coref = None
    self.mention_idx = -1
    self.mention = None
    # surface and lemma forms for embedding lookup
    self.surface_output = ''
    self.lemma_output = ''
    if self.ner in valid_ner:
      self.surface_output = self.word.lower()
      self.lemma_output = self.ner
    elif self.pos[:2] in valid_pos:
      self.surface_output = self.word.lower()
      self.lemma_output = self.lemma.lower()

  def __str__(self):
    return '{}-{}/{}/{}'.format(self.token_idx, self.word, self.lemma, self.pos)

  def print_with_coref(self):
    '''
    result = self.__str__()
    if self.coref_idx != -1:
      result += '-coref#{}-{}'.format(self.coref_idx, self.coref.rep_mention.head_text)
    return result
    '''
    text_form = '_'.join([self.word] + [token.word for token in self.compounds])
    result = '{}-{}'.format(self.token_idx, text_form)
    if self.coref_idx != -1:
      result += '-entity#{:0>3d}'.format(self.coref_idx)
    return result

  def add_idx_info(self, sent_idx, token_idx):
    self.sent_idx = sent_idx
    self.token_idx = token_idx

  def add_coref_info(self, coref_idx, coref, mention_idx, mention):
    assert self.sent_idx == mention.sent_idx and \
        self.token_idx == mention.head_token_idx, \
        'Error! Adding coref info (sent #{}, head_token #{})'.format(
            mention.sent_idx, mention.head_token_idx) + \
                ' to wrong token (sent_idx #{}, toke #{})!'.format(
                    self.sent_idx, self.token_idx)
    assert coref.mentions[mention_idx] == mention, \
        'Error! The {}th mention of {} is not {}'.format(
            mention_idx, coref, mention)

    self.coref_idx = coref_idx
    self.coref = coref
    self.mention_idx = mention_idx
    self.mention = mention

  def get_embedding(self, model, suffix=''):
    if self.pos[:2] in valid_pos:
      try:
        return model.get_embedding(self.word + suffix)
      except KeyError:
        try:
          return model.get_embedding(self.lemma + suffix)
        except KeyError:
          pass
    return None

