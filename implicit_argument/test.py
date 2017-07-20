import pickle as pkl
import sys
from operator import itemgetter
from os.path import join, exists

from imp_arg_reader import read_imp_arg_dataset
from node import SplitNode
from prop_bank_reader import PropBankReader
from ptb_reader import PTBReader
from utils import print_stats
from corenlp_reader import CoreNLPReader


nombank_file = '/Users/pengxiang/corpora/nombank.1.0/nombank.1.0_sorted'
propbank_file = '/Users/pengxiang/corpora/propbank-LDC2004T14/data/prop.txt'

corpus_root = '/Users/pengxiang/corpora/penn-treebank-rel3/parsed/mrg/wsj'
file_pattern = '.*/wsj_.*.mrg'

corenlp_root = '/Users/pengxiang/corpora/wsj_corenlp/20170613'

corenlp_dict_path = './corenlp_dict.pkl'

predicate_dict_path = './predicate_dict.pkl'


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


def build_predicate_dict(annotations):
    predicate_dict = {}
    for predicate in annotations:
        node = str(predicate.node)
        assert node not in predicate_dict
        predicate_dict[node] = predicate.n_pred
    return predicate_dict


def main():
    input_xml = sys.argv[1]

    annotations = read_imp_arg_dataset(input_xml)

    if not exists(corenlp_dict_path):
        print '\nNo existing CoreNLP dict found, reading from source.\n'
        corenlp_reader = CoreNLPReader.build(annotations, corenlp_root)
        corenlp_reader.save(corenlp_dict_path)
    else:
        corenlp_reader = CoreNLPReader.load(corenlp_dict_path)

    ptb_reader = PTBReader(corpus_root, file_pattern)

    if not exists(predicate_dict_path):
        print '\nNo existing predicate dict found, parsing predicate node.\n'

        for predicate in annotations:
            n_pred = corenlp_reader.predicate(predicate.node)
            if n_pred not in nominal_predicate_mapping:
                for subword in n_pred.split('-'):
                    if subword in nominal_predicate_mapping:
                        n_pred = subword
                        break
            predicate.n_pred = n_pred

        predicate_dict = build_predicate_dict(annotations)
        print '\nSaving predicate dict to {}\n'.format(predicate_dict_path)
        pkl.dump(predicate_dict, open(predicate_dict_path, 'w'))
    else:
        print '\nLoading predicate dict from {}\n'.format(predicate_dict_path)

        predicate_dict = pkl.load(open(predicate_dict_path, 'r'))
        for predicate in annotations:
            predicate.n_pred = predicate_dict[str(predicate.node)]

    for predicate in annotations:
        predicate.v_pred = nominal_predicate_mapping[predicate.n_pred]

    propbank_reader = PropBankReader()
    propbank_reader.read_propbank(propbank_file)

    nombank_reader = PropBankReader()
    nombank_reader.read_nombank(nombank_file)

    for predicate in annotations:
        predicate.add_candidates(
            propbank_reader.search_by_file(predicate.node.file_id))
        predicate.add_candidates(
            nombank_reader.search_by_file(predicate.node.file_id))
        predicate.filter_exp_args_from_candidates()
        predicate.check_exp_args()

    for predicate in annotations:
        for label in predicate.imp_args.keys():
            for node in predicate.imp_args[label]:
                ptb_reader.parse_node(node)
                corenlp_reader.parse_node(node)
                # if isinstance(node, SplitNode):
                #     print 'SPLIT: ', node
                # else:
                #     print node
                # print '\t', node.ptb_surface
                # print '\t', node.head_word

    for predicate in annotations:
        for node, _ in predicate.candidates:
            ptb_reader.parse_node(node)
            corenlp_reader.parse_node(node)
            # print node
            # print '\t', node.ptb_surface
            # print '\t', node.head_word

    # for predicate in annotations:
    #     if predicate.n_pred == 'fund':
    #         if predicate.num_imp_arg() > 0:
    #             print '{}\t{}'.format(predicate.node, predicate.n_pred)
    #             for label in predicate.imp_args:
    #                 if predicate.has_oracle(label):
    #                     print '*** implicit {}:'.format(label)
    #                 else:
    #                     print 'implicit {}:'.format(label)
    #                 print '\n'.join(['\t' + str(node) + '\t' + node.ptb_surface
    #                                  for node in predicate.imp_args[label]])
    #             print 'candidates:'
    #             print '\n'.join(['\t' + str(node) + '\t' + node.ptb_surface
    #                              for node, _ in predicate.candidates])
    #             print

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
