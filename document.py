from event_script import EventScript


class Doc:
    def __init__(self):
        self.sents = []
        self.corefs = []
        # event script
        self.script = None
        # word and syntax list
        self.all_list = {}

    def add_sent(self, sent):
        sent.build_dep_graph()
        sent.process_verb_prt()
        self.sents.append(sent)

    def add_coref(self, coref):
        self.corefs.append(coref)

    def __str__(self):
        result = '\t\t#SENT#\t\t'.join([str(sent) for sent in self.sents])
        result += '\t\t\t#DOC#\t\t'
        result += '\t\t#COREF#\t\t'.join([str(coref) for coref in self.corefs])
        return result

    def pretty_print(self):
        result = '\n'.join(['Sent #{}\n\t'.format(sent_idx) + sent.pretty_print()
                            for sent_idx, sent in enumerate(self.sents)])
        result += '\nCoreferences:\n'
        result += '\n'.join(['\t' + str(coref) for coref in self.corefs])
        return result

    def to_plain_text(self):
        return '\n'.join([sent.to_plain_text() for sent in self.sents])

    @classmethod
    def construct(cls, sents, corefs):
        doc = cls()
        for sent in sents:
            doc.add_sent(sent)
        for coref in corefs:
            doc.add_coref(coref)
        return doc

    def fix_ontonotes_coref_info(self):
        for coref in self.corefs:
            # find head token index for every mention
            for mention in coref.mentions:
                mention.head_token_idx = \
                    self.sents[mention.sent_idx].get_head_token_idx(
                        mention.start_token_idx, mention.end_token_idx)
            # find representative mention
            # select mentions headed by proper nouns
            cand_indices = [idx for idx, mention in enumerate(coref.mentions)
                                if self.sents[mention.sent_idx].tokens[mention.head_token_idx].pos.startswith('NNP')]
            # if no mentions are headed by proper nouns, select mentions headed by common nouns
            if not cand_indices:
                cand_indices = [idx for idx, mention in enumerate(coref.mentions)
                                if self.sents[mention.sent_idx].tokens[mention.head_token_idx].pos.startswith('NN')]
            # if no mentions are headed by either proper nouns or common noun, use all mentions as candidates
            if not cand_indices:
                cand_indices = range(0, len(coref.mentions))
            # select from candidate mentions the one with longest string
            cand_length = [len(coref.mentions[idx].text) for idx in cand_indices]
            rep_idx = cand_indices[cand_length.index(max(cand_length))]
            coref.mentions[rep_idx].rep = True
            coref.rep_mention = coref.mentions[rep_idx]

    def preprocessing(self):
        '''
        # build dependency graph for each sentence
        for sent in self.sents:
          sent.build_dep_graph()
          sent.process_verb_prt()
        '''
        for coref_idx, coref in enumerate(self.corefs):
            for mention_idx, mention in enumerate(coref.mentions):
                # add token info to all mentions
                mention.add_token_info(self.sents[mention.sent_idx].tokens)
                # add coref info to all tokens
                mention.head_token.add_coref_info(
                    coref_idx, coref, mention_idx, mention)

    def count_coref_occurrence_in_script(self, only_count_event_arg=False):
        if only_count_event_arg:
            for coref in self.corefs:
                coref.occ_count = 0
            for coref_idx in self.script.get_all_arg_coref_idx():
                self.corefs[coref_idx].occ_count += 1
        else:
            for coref in self.corefs:
                coref.occ_count = len(coref.mentions)

    def validate_script(self, min_num_coref, min_occ_count,
                        only_count_event_arg=False):
        self.count_coref_occurrence_in_script(only_count_event_arg)
        num_valid_coref = sum([1 for coref in self.corefs
                               if coref.occ_count >= min_occ_count])
        return num_valid_coref >= min_num_coref

    def extract_event_script(self):
        self.script = EventScript()
        for sent in self.sents:
            sent.extract_events()
            for event in sent.events:
                self.script.add_event(event)
        for coref in self.corefs:
            self.script.add_coref(coref)
        self.script.sort()

    def eval_most_freq_coref(self):
        return self.script.eval_most_freq_coref()

    def eval_most_sim_arg(self, model, use_other_args=False,
                          syntax_suffix=False, head_only=False, rep_only=False):
        self.script.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
        return self.script.eval_most_sim_arg(model, use_other_args,
                                             syntax_suffix, head_only, rep_only)

    def eval_most_sim_event(self, model, use_max_score=False,
                            syntax_suffix=False, head_only=False, rep_only=False):
        self.script.get_all_embeddings(model, syntax_suffix, head_only, rep_only)
        return self.script.eval_most_sim_event(model, use_max_score,
                                               syntax_suffix, head_only, rep_only)

    '''
    def count_coref_occurrence_in_script(self, only_count_event_arg=False):
      if only_count_event_arg:
        for coref in self.corefs:
          coref.occ_count = 0
        for coref_idx in self.script.get_all_arg_coref_idx():
          self.corefs[coref_idx].occ_count += 1
      else:
        for coref in self.corefs:
          coref.occ_count = len(coref.mentions)

    def validate_script(self, min_num_coref, min_occ_count):
      self.count_coref_occurrence_in_script(True)
      num_valid_coref = sum([1 for coref in self.corefs \
          if coref.occ_count >= min_occ_count])
      return num_valid_coref >= min_num_coref

    def compute_coref_word2vec(self, only_use_rep_mention=False):
      for coref in self.corefs:
        coref.word2vec = np.zeros(300)
        for mention in coref.mentions:
          if not only_use_rep_mention or mention.rep:
            mention.word2vec = np.zeros(300)
            for token_idx in range(mention.start_token_idx, mention.end_token_idx):
              cur_token = self.sents[mention.sent_idx].tokens[token_idx]
              if cur_token.pos.startswith('NN') and cur_token.word2vec is not None:
                mention.word2vec += cur_token.word2vec
            coref.word2vec += mention.word2vec

    def evaluate_most_freq(self, only_count_event_arg=False):
      self.count_coref_occurrence_in_script(only_count_event_arg)
      return self.script.evaluate_most_freq(self.corefs)

    def evaluate_word2vec(
        self, only_use_rep_mention=False, use_other_pred_args=False):
      self.compute_coref_word2vec(only_use_rep_mention)
      return self.script.evaluate_word2vec(
          self.corefs, use_other_pred_args)

    def evaluate_word2vec_script(
        self, only_use_rep_mention=False, use_max_sim_score=False):
      self.compute_coref_word2vec(only_use_rep_mention)
      return self.script.evaluate_word2vec_script(
          self.corefs, use_max_sim_score)
    '''

    def extract_all_list(self):
        self.all_list['surface'] = []
        self.all_list['surface_pair'] = []
        self.all_list['surface_triple'] = []
        self.all_list['lemma'] = []
        self.all_list['lemma_pair'] = []
        self.all_list['lemma_triple'] = []
        for sent in self.sents:
            # sent.process_noun_compounds_and_particles()
            surface_list, lemma_list = sent.produce_token_list()
            surface_pair_list, surface_triple_list, lemma_pair_list, lemma_triple_list = sent.produce_syntax_list()
            self.all_list['surface'].append(surface_list)
            self.all_list['surface_pair'].append(surface_pair_list)
            self.all_list['surface_triple'].append(surface_triple_list)
            self.all_list['lemma'].append(lemma_list)
            self.all_list['lemma_pair'].append(lemma_pair_list)
            self.all_list['lemma_triple'].append(lemma_triple_list)

    def extract_all_pairs(self, word_type, ctx_type, window_size):
        if window_size == 0 or window_size < -1:
            print 'window_size = {0}, not valid, return.'.format(window_size)
            return []
        self.extract_all_list()
        all_pairs = extract_word_ctx_pairs(
            self.all_list[word_type], self.all_list[ctx_type], window_size)
        return all_pairs
