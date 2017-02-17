from eval_instance import EvalInstance

class EmbeddingState:
  def __init__(self):
    self.initiated = False
    self.syntax_suffix = False
    self.head_only = True
    self.rep_only = True

  def set_initiated(self):
    flag = self.initiated
    self.initiated = True
    return flag

  def set_syntax_suffix(self, syntax_suffix):
    flag = (self.syntax_suffix == syntax_suffix)
    self.syntax_suffix = syntax_suffix
    return flag

  def set_head_only(self, head_only):
    flag = (self.head_only == head_only)
    self.head_only = head_only
    return flag

  def set_rep_only(self, rep_only):
    flag = (self.rep_only == rep_only)
    self.rep_only = rep_only
    return flag

  def check_consistency(self, syntax_suffix, head_only, rep_only):
    return self.set_initiated() and \
        self.set_syntax_suffix(syntax_suffix) and \
        self.set_head_only(head_only) and \
        self.set_rep_only(rep_only)

#TODO: whether or not to include negation information in event representation
class Event:
  def __init__(self, pred):
    check_arg_class(pred)
    self.pred = pred
    self.subj = None
    self.obj = None
    self.pobj_list = []
    self.embedding_state = EmbeddingState()
    self.pred_embedding = None
    self.subj_embedding = None
    self.obj_embedding = None
    self.pobj_embedding_list = []
    self.embedding = None
    self.eval_instances = []

  @classmethod
  def construct(cls, pred, subj, obj, pobj_list):
    event = cls(pred);
    if subj is not None:
      event.add_subj(subj)
    if obj is not None:
      event.add_obj(obj)
    for prep, pobj in pobj_list:
      if prep != '' and pobj is not None:
        event.add_pobj(prep, pobj)
    return event

  def get_arg_num(self):
    arg_num = 0
    if self.subj is not None:
      arg_num += 1
    if self.obj is not None:
      arg_num += 1
    for prep, pobj in self.pobj_list:
      if prep != '' and pobj is not None:
        arg_num += 1
    return arg_num

  def get_all_arg(self):
    return [arg for arg in \
        [('SUBJ', self.subj), ('OBJ', self.obj)] + \
        [('PREP_' + prep, pobj) for prep, pobj in self.pobj_list] \
        if arg[1] is not None and arg[1].coref_idx != -1]

  def get_all_arg_coref_idx(self):
    return [arg[1].coref_idx for arg in self.get_all_arg()]

  def idx_key(self):
    return '{:0>4d}{:0>4d}'.format(self.pred.sent_idx, self.pred.token_idx)

  def add_subj(self, subj):
    assert subj is not None, 'Cannot add a None subject!'
    assert self.subj is None, 'Cannot add a subject while it already exists'
    check_arg_class(subj)
    self.subj = subj

  def add_obj(self, obj):
    assert obj is not None, 'Cannot add a None direct object!'
    assert self.obj is None, 'Cannot add an object while it already exists'
    check_arg_class(obj)
    self.obj = obj

  def add_pobj(self, prep, pobj):
    assert prep != '', 'Cannot add a pobj with empty preposition!'
    assert pobj is not None, 'Cannot add a None prepositional object!'
    check_arg_class(pobj)
    self.pobj_list.append((prep, pobj))

  def __str__(self):
    '''
    result = '{}\tSUBJ-{}\tOBJ-{}'.format(
      str(self.pred),
      self.subj.print_with_coref() if self.subj else '',
      self.obj.print_with_coref() if self.obj else '')
    for prep, pobj in self.pobj_list:
      result += '\tPREP_{}-{}'.format(
          prep,
          pobj.print_with_coref() if pobj else '')
    return result
    '''
    result = '{} ( {}, {}'.format(
        str(self.pred),
        self.subj.print_with_coref() if self.subj else '_',
        self.obj.print_with_coref() if self.obj else '_')
    for prep, pobj in self.pobj_list: 
      result += ', {}_{}'.format(
          prep,
          pobj.print_with_coref() if pobj else '_')
    result += ' )'
    return result

  def print_with_idx(self):
    return 'sent#{:0>3d}    {}'.format(
        self.pred.sent_idx, self.__str__())

  def get_pred_embedding(self, model, syntax_suffix=False):
    if syntax_suffix:
      self.pred_embedding = self.pred.get_embedding(model, '-PRED')
    else:
      self.pred_embedding = self.pred.get_embedding(model, '')

  def get_subj_embedding(self, model, syntax_suffix=False, head_only=False, rep_only=False):
    if self.subj is None or self.subj.coref is None:
      self.subj_embedding = None
      return
    if syntax_suffix:
      self.subj_embedding = self.subj.coref.get_embedding(model, '-SUBJ', head_only, rep_only)
    else:
      self.subj_embedding = self.subj.coref.get_embedding(model, '', head_only, rep_only)

  def get_obj_embedding(self, model, syntax_suffix=False, head_only=False, rep_only=False):
    if self.obj is None or self.obj.coref is None:
      self.obj_embedding = None
      return
    if syntax_suffix:
      self.obj_embedding = self.obj.coref.get_embedding(model, '-OBJ', head_only, rep_only)
    else:
      self.obj_embedding = self.obj.coref.get_embedding(model, '', head_only, rep_only)

  def get_pobj_embedding_list(self, model, syntax_suffix=False, head_only=False, rep_only=False):
    self.pobj_embedding_list = [] 
    for prep, pobj in self.pobj_list:
      if pobj.coref is None:
        self.pobj_embedding_list.append(None)
        continue
      if syntax_suffix:
        self.pobj_embedding_list.append(pobj.coref.get_embedding(model, '-PREP_'+prep, head_only, rep_only))
        '''
        # backoff case, seemed of no use under current model
        if embedding is None:
          embedding = pobj.coref.get_embedding(model, '-PREP', head_only, rep_only)
        '''
      else:
        self.pobj_embedding_list.append(pobj.coref.get_embedding(model, '', head_only, rep_only))

    assert len(self.pobj_list) == len(self.pobj_embedding_list), \
        'Number of pobj embeddings mismatch with number of pobjs!'

  def get_all_embeddings(self, model, syntax_suffix=False, head_only=False, rep_only=False):
    # check if current setting match previous settings, if so return previous embedding directly
    if self.embedding_state.check_consistency(syntax_suffix, head_only, rep_only):
      '''
      print 'Embedding already exists with same configuration, ' + \
          'syntax_suffix = {}, head_only = {}, rep_only = {}'.format(
              syntax_suffix, head_only, rep_only)
      '''
      return self.embedding

    '''
    print 'Extract embeddings for predicate and all arguments with ' + \
        'syntax_suffix = {}, head_only = {}, rep_only = {}'.format(
            syntax_suffix, head_only, rep_only)
    '''

    # initialize with zero array
    self.embedding = model.zeros()
    # get predicate embedding
    self.get_pred_embedding(model, syntax_suffix)
    # return None if predicate embedding is None
    if self.pred_embedding is None:
      return
    self.embedding += self.pred_embedding

    # get subject embedding
    self.get_subj_embedding(model, syntax_suffix, head_only, rep_only)
    if self.subj_embedding is not None:
      self.embedding += self.subj_embedding
    # get object embedding
    self.get_obj_embedding(model, syntax_suffix, head_only, rep_only)
    if self.obj_embedding is not None:
      self.embedding += self.obj_embedding
    # get pobj embeddings
    self.get_pobj_embedding_list(model, syntax_suffix, head_only, rep_only)
    for pobj_embedding in self.pobj_embedding_list:
      if pobj_embedding is not None:
        self.embedding += pobj_embedding

    return self.embedding

  def get_eval_instances(self):
    self.eval_instances = []
    if self.pred_embedding is None:
      return self.eval_instances
    idx_label_token_list = []
    if self.subj_embedding is not None:
      idx_label_token_list.append((
        self.subj.coref_idx, 'SUBJ', self.subj_embedding))
    if self.obj_embedding is not None:
      idx_label_token_list.append((
        self.obj.coref_idx, 'OBJ', self.obj_embedding))
    for pobj_idx in range(len(self.pobj_list)):
      if self.pobj_embedding_list[pobj_idx] is not None:
        idx_label_token_list.append((
          self.pobj_list[pobj_idx][1].coref_idx,
          'PREP_' + self.pobj_list[pobj_idx][0],
          self.pobj_embedding_list[pobj_idx]))

    for idx in range(len(idx_label_token_list)):
      self.eval_instances.append(EvalInstance(
        idx_label_token_list[idx][0],
        idx_label_token_list[idx][1],
        self.pred_embedding,
        [embedding for _, _, embedding in \
            idx_label_token_list[:idx] + \
            idx_label_token_list[idx + 1:]]))

    return self.eval_instances

def check_arg_class(arg):
  assert arg.__class__.__name__ == 'Token', \
      'Invalid input argument class {}, only accept Token instance!'.format(
          arg.__class__.__name__)

