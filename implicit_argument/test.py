import pickle as pkl
import sys
from bz2 import BZ2File
from operator import itemgetter
from os.path import join, exists

from corenlp import read_doc_from_corenlp
from imp_arg_reader import read_imp_arg_dataset
from node import SplitNode
from prop_bank_reader import PropBankReader
from ptb_reader import PTBReader
from utils import print_stats


nombank_file = '/Users/pengxiang/corpora/nombank.1.0/nombank.1.0_sorted'
propbank_file = '/Users/pengxiang/corpora/propbank-LDC2004T14/data/prop.txt'

corpus_root = '/Users/pengxiang/corpora/penn-treebank-rel3/parsed/mrg/wsj'
file_pattern = '.*/wsj_.*.mrg'

corenlp_root = '/Users/pengxiang/corpora/penn-treebank-rel3/corenlp/wsj'

corenlp_dict_path = './corenlp_dict.pkl'


nominal_predicate_mapping = {
    'bid': 'bid',
    'sale': 'sell',
    'loan': 'loan',
    'cost': 'cost',
    'plan': 'plan',
    'investor': 'invest',
    'price': 'price',
    'loss': 'lose',
    'investment': 'invest',
    'fund': 'fund',
}


def build_corenlp_dict(annotations):
    corenlp_dict = {}
    for predicate in annotations:
        node = predicate.node
        if node.file_id not in corenlp_dict:
            path = join(corenlp_root, 'parsed', node.directory,
                        node.file_id + '.xml.bz2')
            doc = read_doc_from_corenlp(BZ2File(path, 'r'))
            path = join(corenlp_root, 'idx', node.directory, node.file_id)
            idx_mapping = []
            with open(path, 'r') as fin:
                for line in fin:
                    idx_mapping.append([int(i) for i in line.split()])
            corenlp_dict[node.file_id] = (doc, idx_mapping)
    return corenlp_dict


def main():
    input_xml = sys.argv[1]

    annotations = read_imp_arg_dataset(input_xml)

    if not exists(corenlp_dict_path):
        print '\nNo existing CoreNLP dict found, reading from source.\n'
        corenlp_dict = build_corenlp_dict(annotations)
        print '\nSaving CoreNLP dict to {}\n'.format(corenlp_dict_path)
        pkl.dump(corenlp_dict, open(corenlp_dict_path, 'w'))
    else:
        print '\nLoading CoreNLP dict from {}\n'.format(corenlp_dict_path)
        corenlp_dict = pkl.load(open(corenlp_dict_path, 'r'))

    ptb_reader = PTBReader(corpus_root, file_pattern, corenlp_dict)

    propbank_reader = PropBankReader()
    propbank_reader.read_propbank(propbank_file)

    nombank_reader = PropBankReader()
    nombank_reader.read_nombank(nombank_file)

    for predicate in annotations:
        predicate.add_candidates(
            propbank_reader.search_by_file(predicate.node.file_id))
        predicate.add_candidates(
            nombank_reader.search_by_file(predicate.node.file_id))
        # predicate.filterExpArgsFromCandidates()
        # predicate.checkExpArgs()

    for predicate in annotations:

        n_pred = ptb_reader.parse_single_node(predicate.node)
        if n_pred not in nominal_predicate_mapping:
            for subword in n_pred.split('-'):
                if subword in nominal_predicate_mapping:
                    n_pred = subword
                    break
        predicate.n_pred = n_pred
        predicate.v_pred = nominal_predicate_mapping[n_pred]

        for label in predicate.imp_args.keys():
            for node in predicate.imp_args[label]:
                ptb_reader.parse_node(node)
                # if isinstance(node, SplitNode):
                #     print 'SPLIT: ', node
                # else:
                #     print node
                # print '\t', node.ptb_surface
                # print '\t', node.head_word

        for node, _ in predicate.candidates:
            ptb_reader.parse_node(node)
            # print node
            # print '\t', node.ptb_surface
            # print '\t', node.head_word

    for predicate in annotations:
        if predicate.n_pred == 'fund':
            if predicate.num_imp_arg() > 0:
                print '{}\t{}'.format(predicate.node, predicate.n_pred)
                for label in predicate.imp_args:
                    if predicate.has_oracle(label):
                        print '*** implicit {}:'.format(label)
                    else:
                        print 'implicit {}:'.format(label)
                    print '\n'.join(['\t' + str(node) + '\t' + node.ptb_surface
                                     for node in predicate.imp_args[label]])
                print 'candidates:'
                print '\n'.join(['\t' + str(node) + '\t' + node.ptb_surface
                                 for node, _ in predicate.candidates])
                print
    # for predicate in annotations:
    #     if predicate.n_pred == 'investment':
    #         print '{}\t{}'.format(predicate.node, predicate.n_pred)
    #         if predicate.imp_args:
    #             for label in predicate.imp_args:
    #                 print 'implicit {}:'.format(label)
    #                 for node in predicate.imp_args[label]:
    #                     print '\t{}{}'.format(
    #                         str(node),
    #                         '\t(incorporated)'
    #                         if ptb_reader.is_child_node(node, predicate.node)
    #                         else '')
    #         if predicate.exp_args:
    #             for label in predicate.exp_args:
    #                 print label
    #                 for node in predicate.exp_args[label]:
    #                     print '\t' + str(node)

    if len(sys.argv) > 2:
        pkl.dump(annotations, open(sys.argv[2], 'w'))

    print_stats(annotations)

if __name__ == '__main__':
    main()
