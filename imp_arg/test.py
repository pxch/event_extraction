import pickle as pkl
import sys
from os.path import exists

from tqdm import tqdm

from candidate import CandidateDict
from consts import predicate_dict_path, corenlp_dict_path
from corpus_reader import CoreNLPReader
from corpus_reader import NombankReader, PropbankReader
from corpus_reader import TreebankReader
from implicit_argument_instance import ImplicitArgumentInstance
from predicate import Predicate
from stats import print_stats

max_candidate_dist = 2


def read_imp_arg_dataset(file_path):

    print '\nReading implicit argument dataset from {}'.format(file_path)
    input_xml = open(file_path, 'r')

    all_instances = []
    for line in input_xml.readlines()[1:-1]:
        instance = ImplicitArgumentInstance.parse(line.strip())
        all_instances.append(instance)

    print '\tFound {} instances'.format(len(all_instances))

    all_instances.sort(key=lambda ins: str(ins.pred_pointer))

    return all_instances


def main():

    all_instances = read_imp_arg_dataset(sys.argv[1])

    # fout = open('test_annotations.txt', 'w')
    # fout.write('<annotations>\n')
    # for instance in all_instances:
    #     fout.write(str(instance) + '\n')
    # fout.write('</annotations>\n')
    # fout.close()

    print '\nLoading predicate dict from {}'.format(predicate_dict_path)
    predicate_dict = pkl.load(open(predicate_dict_path, 'r'))

    # all_instances_by_pred = defaultdict(list)
    # for instance in all_instances:
    #     n_pred = predicate_dict[str(instance.pred_pointer)]
    #     all_instances_by_pred[n_pred].append(instance)
    #
    # for n_pred in all_instances_by_pred:
    #     fout = open(join('ia_by_pred', n_pred), 'w')
    #     for instance in all_instances_by_pred[n_pred]:
    #         fout.write(str(instance) + '\n')
    #     fout.close()

    all_predicates = []
    for instance in all_instances:
        predicate = Predicate.build(instance)
        predicate.set_pred(predicate_dict[str(predicate.pred_pointer)])
        all_predicates.append(predicate)

    treebank_reader = TreebankReader()

    propbank_reader = PropbankReader()
    propbank_reader.build_index()

    nombank_reader = NombankReader()
    nombank_reader.build_index()

    if not exists(corenlp_dict_path):
        print '\nNo existing CoreNLP dict found'
        corenlp_reader = CoreNLPReader.build(all_predicates)
        corenlp_reader.save(corenlp_dict_path)
    else:
        corenlp_reader = CoreNLPReader.load(corenlp_dict_path)

    print '\nChecking explicit arguments with Nombank instances'
    for predicate in all_predicates:
        nombank_instance = nombank_reader.search_by_pointer(
            predicate.pred_pointer)
        predicate.check_exp_args(nombank_instance, add_missing_args=False,
                                 remove_conflict_imp_args=False, verbose=False)

    print '\nParsing all implicit and explicit arguments'
    for predicate in tqdm(all_predicates, desc='Processed', ncols=100):
        predicate.parse_args(treebank_reader, corenlp_reader)

    print '\nAdding candidates from Propbank and Nombank'
    candidate_dict = CandidateDict(propbank_reader, nombank_reader,
                                   corenlp_reader, max_dist=max_candidate_dist)
    for predicate in tqdm(all_predicates, desc='Processed', ncols=100):
        for candidate in candidate_dict.get_candidates(predicate.pred_pointer):
            predicate.candidates.append(candidate)

    # fout = open('all_predicates.txt', 'w')
    # for predicate in all_predicates:
    #     fout.write('{}\t{}\n'.format(
    #         predicate.pred_pointer, predicate.n_pred))
    #     for label, fillers in predicate.imp_args.items():
    #         fout.write('\tImplicit {}:\n'.format(label))
    #         for filler in fillers:
    #             fout.write('\t\t{}\n'.format(filler.pretty_print()))
    #     for label, fillers in predicate.exp_args.items():
    #         fout.write('\tExplicit {}:\n'.format(label))
    #         if len(fillers) > 1:
    #             fout.write('\tMORE THAN ONE FILLER\n')
    #         for filler in fillers:
    #             fout.write('\t\t{}\n'.format(filler.pretty_print()))
    # fout.close()
    #
    # fout = open('all_candidates.txt', 'w')
    # for key, candidates in candidate_dict:
    #     fout.write(key + '\n')
    #     for candidate in candidates:
    #         fout.write('\t{}\t{}\n'.format(
    #             candidate.arg_label, candidate.arg_pointer.pretty_print()))
    # fout.close()

    print '\nPrinting statistics'
    print_stats(all_predicates)


main()
