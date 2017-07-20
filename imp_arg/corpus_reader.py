import pickle as pkl
from bz2 import BZ2File
from collections import defaultdict
from os.path import join

import nltk_corpus
from corenlp import read_doc_from_corenlp
from rich_script import RichScript, Script
from util import read_vocab_list

nombank_function_tag_mapping = {
    'TMP': 'temporal',
    'LOC': 'location',
    'MNR': 'manner',
    'PNC': 'purpose',
    'NEG': 'negation',
    'EXT': 'extent',
    'ADV': 'adverbial',
    # tags below do not appear in implicit argument dataset
    'DIR': 'directional',
    'PRD': 'predicative',
    'CAU': 'cause',
    'DIS': 'discourse',
    'REF': 'reference'
}


def convert_nombank_label(label):
    if label[:3] == 'ARG':
        if label[3].isdigit():
            return label[:4].lower()
        elif label[3] == 'M':
            function_tag = label.split('-')[1]
            return nombank_function_tag_mapping.get(function_tag, '')
    return ''


core_arg_list = ['arg0', 'arg1', 'arg2', 'arg3', 'arg4']


def is_core_arg(label):
    return label in core_arg_list


corenlp_root = '/Users/pengxiang/corpora/wsj_corenlp/20170613'
prep_vocab_list_file = '../vocab_list/preposition'


class TreebankReader(object):
    def __init__(self):
        print '\nBuilding TreebankReader from {}'.format(nltk_corpus.treebank_root)
        self.treebank = nltk_corpus.treebank
        print '\tFound {} files'.format(len(self.treebank.fileids()))

        self.all_sents = []
        self.all_tagged_sents = []
        self.all_parsed_sents = []
        self.treebank_fileid = ''

    def read_file(self, treebank_fileid):
        if treebank_fileid != self.treebank_fileid:
            self.all_sents = self.treebank.sents(fileids=treebank_fileid)
            self.all_tagged_sents = \
                self.treebank.tagged_sents(fileids=treebank_fileid)
            self.all_parsed_sents = \
                self.treebank.parsed_sents(fileids=treebank_fileid)
            self.treebank_fileid = treebank_fileid


class BaseCorpusReader(object):
    def __init__(self, instances):
        self.instances = instances
        self.instances_by_fileid = defaultdict(list)

    @staticmethod
    def convert_fileid(fileid):
        # result = re.sub(r'^\d\d/', '', fileid)
        # result = re.sub(r'\.mrg$', '', result)
        # return result
        return fileid[3:11]

    def build_index(self):
        print '\tBuilding index by fileid'
        for instance in self.instances:
            fileid = BaseCorpusReader.convert_fileid(instance.fileid)
            self.instances_by_fileid[fileid].append(instance)

    def search_by_fileid(self, fileid):
        return self.instances_by_fileid.get(fileid, [])

    def search_by_pointer(self, pointer):
        for instance in self.search_by_fileid(pointer.fileid):
            if instance.sentnum == pointer.sentnum \
                    and instance.wordnum == pointer.tree_pointer.wordnum:
                return instance
        return None


class PropbankReader(BaseCorpusReader):
    def __init__(self):
        print '\nBuilding PropbankReader from {}/{}'.format(
            nltk_corpus.propbank_root, nltk_corpus.propbank_file)
        super(PropbankReader, self).__init__(nltk_corpus.propbank.instances())

        self.num_instances = len(self.instances)
        print '\tFound {} instances'.format(self.num_instances)


class NombankReader(BaseCorpusReader):
    def __init__(self):
        print '\nBuilding NombankReader from {}/{}'.format(
            nltk_corpus.nombank_root, nltk_corpus.nombank_file)
        super(NombankReader, self).__init__(nltk_corpus.nombank.instances())

        self.num_instances = len(self.instances)
        print '\tFound {} instances'.format(self.num_instances)


class CoreNLPReader(object):
    def __init__(self, corenlp_dict):
        self.corenlp_dict = corenlp_dict

    def get_all(self, fileid):
        return self.corenlp_dict[fileid]

    def get_idx_mapping(self, fileid):
        return self.corenlp_dict[fileid][0]

    def get_doc(self, fileid):
        return self.corenlp_dict[fileid][1]

    def get_script(self, fileid):
        return self.corenlp_dict[fileid][2]

    def get_rich_script(self, fileid):
        return self.corenlp_dict[fileid][3]

    @classmethod
    def build(cls, predicates):
        prep_vocab_list = read_vocab_list(prep_vocab_list_file)

        print '\nBuilding CoreNLP Reader from {}'.format(corenlp_root)
        corenlp_dict = {}

        for predicate in predicates:
            pred_pointer = predicate.pred_pointer
            if pred_pointer.fileid not in corenlp_dict:

                path = join(corenlp_root, 'idx', pred_pointer.get_path())
                idx_mapping = []
                with open(path, 'r') as fin:
                    for line in fin:
                        idx_mapping.append([int(i) for i in line.split()])

                path = join(corenlp_root, 'parsed',
                            pred_pointer.get_path('.xml.bz2'))
                doc = read_doc_from_corenlp(BZ2File(path, 'r'))

                script = Script.from_doc(doc)

                rich_script = RichScript.build(
                    script,
                    prep_vocab_list=prep_vocab_list,
                    use_lemma=True,
                    filter_stop_events=False
                )

                corenlp_dict[pred_pointer.fileid] = \
                    (idx_mapping, doc, script, rich_script)

        return cls(corenlp_dict)

    @classmethod
    def load(cls, corenlp_dict_path):
        print '\nLoading CoreNLP Reader from {}'.format(corenlp_dict_path)
        corenlp_dict = pkl.load(open(corenlp_dict_path, 'r'))
        return cls(corenlp_dict)

    def save(self, corenlp_dict_path):
        print '\nSaving CoreNLP dict to {}'.format(corenlp_dict_path)
        pkl.dump(self.corenlp_dict, open(corenlp_dict_path, 'w'))
