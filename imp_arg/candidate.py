from collections import defaultdict

from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer

from consts import convert_fileid, convert_nombank_label
from consts import core_arg_list
from rich_tree_pointer import RichTreePointer


class CandidateDict(object):
    def __init__(self, propbank_reader, nombank_reader, corenlp_reader,
                 max_dist=2):
        self.propbank_reader = propbank_reader
        self.nombank_reader = nombank_reader
        self.corenlp_reader = corenlp_reader
        self.max_dist = max_dist
        self.candidate_dict = defaultdict(list)

    def __iter__(self):
        for key, candidates in self.candidate_dict.items():
            yield key, candidates

    def get_candidates(self, pred_pointer):
        candidates = []

        fileid = pred_pointer.fileid
        instances = []
        instances.extend(self.propbank_reader.search_by_fileid(fileid))
        instances.extend(self.nombank_reader.search_by_fileid(fileid))

        for sentnum in range(max(0, pred_pointer.sentnum - self.max_dist),
                             pred_pointer.sentnum):
            key = '{}:{}'.format(fileid, sentnum)
            if key not in self.candidate_dict:
                self.add_candidates(key, instances)
            candidates.extend(self.candidate_dict[key])

        key = '{}:{}'.format(fileid, pred_pointer.sentnum)
        if key not in self.candidate_dict:
            self.add_candidates(key, instances)
        for candidate in self.candidate_dict[key]:
            if candidate.pred_pointer != pred_pointer:
                candidates.append(candidate)

        return candidates

    def add_candidates(self, key, instances):
        assert key not in self.candidate_dict
        for instance in instances:
            assert convert_fileid(instance.fileid) == key.split(':')[0]
            if instance.sentnum == int(key.split(':')[1]):
                for candidate in Candidate.from_instance(instance):
                    candidate.arg_pointer.get_corenlp(self.corenlp_reader)
                    candidate.arg_pointer.parse_corenlp()
                    if candidate.arg_pointer.corenlp_word_surface != '':
                        self.candidate_dict[key].append(candidate)


class Candidate(object):
    def __init__(self, fileid, sentnum, pred, arg, arg_label, tree):
        self.fileid = fileid
        self.sentnum = sentnum

        self.pred_pointer = RichTreePointer(fileid, sentnum, pred, tree=tree)
        # self.pred_pointer.parse_treebank()

        self.arg_pointer = RichTreePointer(fileid, sentnum, arg, tree=tree)
        self.arg_pointer.parse_treebank()

        self.arg_label = arg_label

    @staticmethod
    def from_instance(instance):
        candidate_list = []

        fileid = convert_fileid(instance.fileid)
        sentnum = instance.sentnum
        pred = instance.predicate
        tree = instance.tree

        for arg_pointer, label in instance.arguments:
            cvt_label = convert_nombank_label(label)
            if cvt_label in core_arg_list:
                if isinstance(arg_pointer, NombankChainTreePointer) or \
                        isinstance(arg_pointer, PropbankChainTreePointer):
                    for p in arg_pointer.pieces:
                        candidate_list.append(Candidate(
                            fileid, sentnum, pred, p, cvt_label, tree))
                else:
                    candidate_list.append(Candidate(
                        fileid, sentnum, pred, arg_pointer, cvt_label, tree))

        return candidate_list

    def is_oracle(self, imp_args):
        assert self.arg_pointer.has_corenlp_info()
        assert all(arg.has_corenlp_info() for arg in imp_args)

        # if candidate has the same pointer
        if self.arg_pointer in imp_args:
            return True

        # if candidate has the same lemmas from CoreNLP document
        if self.arg_pointer.corenlp_lemma_surface in \
                [arg.corenlp_lemma_surface for arg in imp_args]:
            return True

        # if candidate has the pointer consisting of one imp_arg pointer
        # plus one preceding preposition
        if not self.arg_pointer.is_split_pointer:
            for arg in imp_args:
                if not arg.is_split_pointer:
                    if self.arg_pointer.sentnum == arg.sentnum:
                        cand_tb = self.arg_pointer.treebank_info_list[0]
                        arg_tb = arg.treebank_info_list[0]
                        if cand_tb.eq_with_preceding_prep(arg_tb):
                            return True

        return False

    def dice_score(self, imp_args, use_corenlp_tokens=True):
        dice_score = 0.0

        if len(imp_args) > 0:
            dice_score_list = []
            for arg in imp_args:
                dice_score_list.append(
                    self.arg_pointer.dice_score(
                        arg, use_corenlp_tokens=use_corenlp_tokens))

            dice_score = max(dice_score_list)

        return dice_score
