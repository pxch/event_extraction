import pickle as pkl
import sys
import timeit
from collections import defaultdict
from os.path import exists, join

from tqdm import tqdm

from candidate import CandidateDict
from consts import *
from corpus_reader import CoreNLPReader, NombankReader, PropbankReader
from corpus_reader import TreebankReader
from implicit_argument_instance import ImplicitArgumentInstance
from predicate import Predicate
from rich_predicate import RichPredicate
from stats import print_stats
from util import Word2VecModel, get_class_name

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


def print_imp_arg_dataset(all_instances, file_path):
    fout = open(file_path, 'w')
    fout.write('<annotations>\n')

    for instance in all_instances:
        fout.write(str(instance) + '\n')

    fout.write('</annotations>\n')
    fout.close()


def print_imp_arg_dataset_by_pred(all_instances, predicate_dict, dir_path):
    all_instances_by_pred = defaultdict(list)
    for instance in all_instances:
        n_pred = predicate_dict[str(instance.pred_pointer)]
        all_instances_by_pred[n_pred].append(instance)

    for n_pred in all_instances_by_pred:
        fout = open(join(dir_path, n_pred), 'w')
        for instance in all_instances_by_pred[n_pred]:
            fout.write(str(instance) + '\n')
        fout.close()


def check_multi_pobj(all_predicates):
    for predicate in all_predicates:
        flag = False
        count = 0
        for label in predicate.imp_args.keys():
            if label in core_arg_list and \
                    predicate_core_arg_mapping[
                        predicate.v_pred][label].startswith('PREP'):
                count += 1
                break
        for label in predicate.exp_args.keys():
            if label in core_arg_list and \
                    predicate_core_arg_mapping[
                        predicate.v_pred][label].startswith('PREP'):
                count += 1
        if count > 1:
            flag = True
        if flag:
            print predicate.pretty_print(
                verbose=False, include_candidates=False,
                include_dice_score=False)


def print_all_predicate(
        all_predicates, file_path, verbose=True,include_candidates=True,
        include_dice_scores=True, corenlp_reader=None):

    fout = open(file_path, 'w')

    for predicate in all_predicates:
        fout.write(predicate.pretty_print(
            verbose=verbose,
            include_candidates=include_candidates,
            include_dice_score=include_dice_scores,
            corenlp_reader=corenlp_reader))

    fout.close()


def build_pred_wv_mapping(pred_list, model):
    assert isinstance(model, Word2VecModel), \
        'model must be a {} instance'.format(get_class_name(Word2VecModel))

    pred_wv_mapping = {}
    for pred in pred_list:
        index = model.get_word_index(pred + '-PRED')
        assert index != -1
        pred_wv_mapping[pred] = index

    return pred_wv_mapping


def main():

    all_instances = read_imp_arg_dataset(sys.argv[1])

    # print_imp_arg_dataset(all_instances, 'test_annotations.txt')

    print '\nLoading predicate dict from {}'.format(predicate_dict_path)
    predicate_dict = pkl.load(open(predicate_dict_path, 'r'))

    # print_imp_arg_dataset_by_pred(all_instances, predicate_dict, 'ia_by_pred')

    treebank_reader = TreebankReader()

    propbank_reader = PropbankReader()
    propbank_reader.build_index()

    nombank_reader = NombankReader()
    nombank_reader.build_index()

    if not exists(corenlp_dict_path):
        print '\nNo existing CoreNLP dict found'
        corenlp_reader = CoreNLPReader.build(all_instances)
        corenlp_reader.save(corenlp_dict_path)
    else:
        corenlp_reader = CoreNLPReader.load(corenlp_dict_path)

    if not exists(all_predicates_path):
        all_predicates = []
        for instance in all_instances:
            predicate = Predicate.build(instance)
            predicate.set_pred(predicate_dict[str(predicate.pred_pointer)])
            all_predicates.append(predicate)

        print '\nChecking explicit arguments with Nombank instances'
        for predicate in all_predicates:
            nombank_instance = nombank_reader.search_by_pointer(
                predicate.pred_pointer)
            predicate.check_exp_args(
                nombank_instance, add_missing_args=False,
                remove_conflict_imp_args=False, verbose=False)

        print '\nParsing all implicit and explicit arguments'
        for predicate in tqdm(all_predicates, desc='Processed', ncols=100):
            predicate.parse_args(treebank_reader, corenlp_reader)

        print '\nSaving all parsed predicates to {}'.format(
            all_predicates_path)
        pkl.dump(all_predicates, open(all_predicates_path, 'w'))

    else:
        print '\nLoading all parsed predicates from {}'.format(
            all_predicates_path)

        start_time = timeit.default_timer()
        all_predicates = pkl.load(open(all_predicates_path, 'r'))
        elapsed = timeit.default_timer() - start_time
        print '\tDone in {:.3f} seconds'.format(elapsed)

    # check_multi_pobj(all_predicates)

    if not exists(candidate_dict_path):
        print '\nBuilding candidate dict from Propbank and Nombank'
        candidate_dict = CandidateDict(
            propbank_reader=propbank_reader, nombank_reader=nombank_reader,
            corenlp_reader=corenlp_reader, max_dist=max_candidate_dist)

        for predicate in tqdm(all_predicates, desc='Processed', ncols=100):
            candidate_dict.add_candidates(predicate.pred_pointer)

        candidate_dict.save(candidate_dict_path)

    else:
        candidate_dict = CandidateDict.load(
            candidate_dict_path, propbank_reader=propbank_reader,
            nombank_reader=nombank_reader, corenlp_reader=corenlp_reader,
            max_dist=max_candidate_dist)

    # candidate_dict.print_all_candidates('all_candidates.txt')

    print '\nAdding candidates to predicates'
    for predicate in all_predicates:
        for candidate in candidate_dict.get_candidates(predicate.pred_pointer):
                predicate.candidates.append(candidate)

    # print_all_predicate(all_predicates, 'all_predicates.txt', verbose=True,
    #                     include_candidates=True, include_dice_scores=True,
    #                     corenlp_reader=corenlp_reader)

    if not exists(all_rich_predicates_path):
        all_rich_predicates = []

        for predicate in all_predicates:
            rich_predicate = RichPredicate.build(
                predicate, corenlp_reader, use_lemma=True, use_entity=True,
                use_corenlp_tokens=True)
            all_rich_predicates.append(rich_predicate)

        print '\nSaving all rich predicates to {}'.format(
            all_rich_predicates_path)
        pkl.dump(all_rich_predicates, open(all_rich_predicates_path, 'w'))

    print '\nPrinting statistics'
    print_stats(all_predicates)


main()
