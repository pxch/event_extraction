import pickle as pkl
import timeit
from collections import defaultdict
from os.path import exists, join

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

import consts
from candidate import CandidateDict
from corpus_reader import CoreNLPReader
from corpus_reader import NombankReader, PropbankReader, TreebankReader
from implicit_argument_instance import ImplicitArgumentInstance
from predicate import Predicate
from rich_predicate import RichPredicate
from rich_script import IndexedEvent
from stats import print_stats, print_eval_stats, print_eval_results
from util import Word2VecModel, get_class_name


def build_pred_wv_mapping(pred_list, model):
    assert isinstance(model, Word2VecModel), \
        'model must be a {} instance'.format(get_class_name(Word2VecModel))

    pred_wv_mapping = {}
    for pred in pred_list:
        index = model.get_word_index(pred + '-PRED')
        assert index != -1
        pred_wv_mapping[pred] = index

    return pred_wv_mapping


class ImplicitArgumentReader(object):
    def __init__(self, n_splits=10, max_candidate_dist=2):
        self.n_splits = n_splits
        self.max_candidate_dist = max_candidate_dist

        self.all_instances = []
        self.instance_order_list = []
        self.train_test_folds = []

        self.all_predicates = []
        self.all_rich_predicates = []

        self._treebank_reader = None
        self._nombank_reader = None
        self._propbank_reader = None
        self._predicate_dict = None
        self._corenlp_reader = None
        self._candidate_dict = None

    @property
    def treebank_reader(self):
        if self._treebank_reader is None:
            self._treebank_reader = TreebankReader()
        return self._treebank_reader

    @property
    def nombank_reader(self):
        if self._nombank_reader is None:
            self._nombank_reader = NombankReader()
            self._nombank_reader.build_index()
        return self._nombank_reader

    @property
    def propbank_reader(self):
        if self._propbank_reader is None:
            self._propbank_reader = PropbankReader()
            self._propbank_reader.build_index()
        return self._propbank_reader

    @property
    def predicate_dict(self):
        if self._predicate_dict is None:
            self.load_predicate_dict()
        return self._predicate_dict

    def load_predicate_dict(
            self, predicate_dict_path=consts.predicate_dict_path):
        print '\nLoading predicate dict from {}'.format(predicate_dict_path)
        self._predicate_dict = pkl.load(open(predicate_dict_path, 'r'))

    @property
    def corenlp_reader(self):
        if self._corenlp_reader is None:
            self.load_corenlp_reader()
        return self._corenlp_reader

    def load_corenlp_reader(self, corenlp_dict_path=consts.corenlp_dict_path):
        if not exists(corenlp_dict_path):
            assert len(self.all_instances) > 0
            print '\nNo existing CoreNLP dict found'
            self._corenlp_reader = CoreNLPReader.build(self.all_instances)
            self._corenlp_reader.save(corenlp_dict_path)
        else:
            self._corenlp_reader = CoreNLPReader.load(corenlp_dict_path)

    @property
    def candidate_dict(self):
        if self._candidate_dict is None:
            self.load_candidate_dict()
        return self._candidate_dict

    def load_candidate_dict(
            self, candidate_dict_path=consts.candidate_dict_path):
        if not exists(candidate_dict_path):
            assert len(self.all_predicates) > 0
            print '\nBuilding candidate dict from Propbank and Nombank'
            self._candidate_dict = CandidateDict(
                propbank_reader=self.propbank_reader,
                nombank_reader=self.nombank_reader,
                corenlp_reader=self.corenlp_reader,
                max_dist=self.max_candidate_dist)

            for predicate in tqdm(
                    self.all_predicates, desc='Processed', ncols=100):
                self.candidate_dict.add_candidates(predicate.pred_pointer)

            self._candidate_dict.save(candidate_dict_path)

        else:
            self._candidate_dict = CandidateDict.load(
                candidate_dict_path,
                propbank_reader=self.propbank_reader,
                nombank_reader=self.nombank_reader,
                corenlp_reader=self.corenlp_reader,
                max_dist=self.max_candidate_dist)

    def read_dataset(self, file_path):
        print '\nReading implicit argument dataset from {}'.format(file_path)
        input_xml = open(file_path, 'r')

        all_instances = []
        for line in input_xml.readlines()[1:-1]:
            instance = ImplicitArgumentInstance.parse(line.strip())
            all_instances.append(instance)

        print '\tFound {} instances'.format(len(all_instances))

        self.all_instances = sorted(
            all_instances, key=lambda ins: str(ins.pred_pointer))

        # self.instance_order_list = []
        # for instance in all_instances:
        #     for idx, ins in enumerate(self.all_instances):
        #         if str(instance.pred_pointer) == str(ins.pred_pointer):
        #             self.instance_order_list.append(idx)
        #             break

        self.instance_order_list = [self.all_instances.index(instance)
                                    for instance in all_instances]

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        self.train_test_folds = list(kf.split(self.instance_order_list))

    def print_dataset(self, file_path):
        fout = open(file_path, 'w')
        fout.write('<annotations>\n')

        for instance in self.all_instances:
            fout.write(str(instance) + '\n')

        fout.write('</annotations>\n')
        fout.close()

    def print_dataset_by_pred(self, dir_path):
        all_instances_by_pred = defaultdict(list)
        for instance in self.all_instances:
            n_pred = self.predicate_dict[str(instance.pred_pointer)]
            all_instances_by_pred[n_pred].append(instance)

        for n_pred in all_instances_by_pred:
            fout = open(join(dir_path, n_pred), 'w')
            for instance in all_instances_by_pred[n_pred]:
                fout.write(str(instance) + '\n')
            fout.close()

    def build_predicates(self, all_predicates_path=None):
        assert len(self.all_instances) > 0
        assert len(self.all_predicates) == 0
        for instance in self.all_instances:
            predicate = Predicate.build(instance)
            predicate.set_pred(self.predicate_dict[str(predicate.pred_pointer)])
            self.all_predicates.append(predicate)

        print '\nChecking explicit arguments with Nombank instances'
        for predicate in self.all_predicates:
            nombank_instance = self.nombank_reader.search_by_pointer(
                predicate.pred_pointer)
            predicate.check_exp_args(
                nombank_instance, add_missing_args=False,
                remove_conflict_imp_args=False, verbose=False)

        print '\nParsing all implicit and explicit arguments'
        for predicate in tqdm(self.all_predicates, desc='Processed', ncols=100):
            predicate.parse_args(self.treebank_reader, self.corenlp_reader)

        if all_predicates_path:
            print '\nSaving all parsed predicates to {}'.format(
                all_predicates_path)
            pkl.dump(self.all_predicates, open(all_predicates_path, 'w'))

    def load_predicates(self, all_predicates_path=consts.all_predicates_path):
        print '\nLoading all parsed predicates from {}'.format(
            all_predicates_path)

        start_time = timeit.default_timer()
        self.all_predicates = pkl.load(open(all_predicates_path, 'r'))
        elapsed = timeit.default_timer() - start_time
        print '\tDone in {:.3f} seconds'.format(elapsed)

    def add_candidates(self):
        print '\nAdding candidates to predicates'
        for predicate in self.all_predicates:
            for candidate in self.candidate_dict.get_candidates(
                    predicate.pred_pointer):
                predicate.candidates.append(candidate)

    def print_stats(self):
        print '\nPrinting statistics'
        print_stats(self.all_predicates)

    def build_rich_predicates(self, use_corenlp_token=True,
                              all_rich_predicates_path=None):
        assert len(self.all_predicates) > 0
        assert len(self.all_rich_predicates) == 0

        for predicate in self.all_predicates:
            rich_predicate = RichPredicate.build(
                predicate,
                corenlp_reader=self.corenlp_reader,
                use_lemma=True,
                use_entity=True,
                use_corenlp_tokens=use_corenlp_token)
            self.all_rich_predicates.append(rich_predicate)

        if all_rich_predicates_path:
            print '\nSaving all rich predicates to {}'.format(
                all_rich_predicates_path)
            pkl.dump(self.all_rich_predicates,
                     open(all_rich_predicates_path, 'w'))

    def load_rich_predicates(
            self, all_rich_predicates_path=consts.all_rich_predicates_path):
        print '\nLoading all rich predicates from {}'.format(
            all_rich_predicates_path)

        start_time = timeit.default_timer()
        self.all_rich_predicates = pkl.load(open(all_rich_predicates_path, 'r'))
        elapsed = timeit.default_timer() - start_time
        print '\tDone in {:.3f} seconds'.format(elapsed)

    def compute_coherence_score(self, event_comp_model, use_max_score=True):
        assert len(self.all_rich_predicates) > 0

        word2vec_model = event_comp_model.word2vec

        coherence_fn = event_comp_model.pair_composition_network.coherence_fn
        use_salience = event_comp_model.pair_composition_network.use_salience
        salience_features = \
            event_comp_model.pair_composition_network.salience_features

        context_input_list_mapping = {}

        for rich_predicate in self.all_rich_predicates:
            if rich_predicate.fileid not in context_input_list_mapping:
                rich_script = self.corenlp_reader.get_rich_script(
                    rich_predicate.fileid)
                rich_script.get_index(
                    word2vec_model, include_type=True, use_unk=True)

                rich_event_list = rich_script.get_indexed_events()

                context_input_list = \
                    [rich_event.get_pos_input(include_all_pobj=False)
                     for rich_event in rich_event_list]

                context_input_list_mapping[rich_predicate.fileid] = \
                    context_input_list

        pred_wv_mapping = \
            build_pred_wv_mapping(consts.pred_list, word2vec_model)

        for rich_predicate in self.all_rich_predicates:
            rich_predicate.set_pred_wv(pred_wv_mapping)
            rich_predicate.get_index(
                word2vec_model, include_type=True, use_unk=True)

        exclude_pred_idx_list = []

        pred_idx = 0
        for rich_predicate in tqdm(
                self.all_rich_predicates, desc='Processed', ncols=100):
            if len(rich_predicate.imp_args) == 0:
                continue

            context_input_list = context_input_list_mapping[
                rich_predicate.fileid]
            num_context = len(context_input_list)

            if num_context == 0:
                exclude_pred_idx_list.append(pred_idx)
                continue
            pred_idx += 1

            pred_input_a = np.zeros(num_context, dtype=np.int32)
            subj_input_a = np.zeros(num_context, dtype=np.int32)
            obj_input_a = np.zeros(num_context, dtype=np.int32)
            pobj_input_a = np.zeros(num_context, dtype=np.int32)
            for context_idx, context_input in enumerate(context_input_list):
                assert isinstance(context_input, IndexedEvent), \
                    'context_input must be a {} instance'.format(
                        get_class_name(IndexedEvent))
                pred_input_a[context_idx] = context_input.pred_input
                subj_input_a[context_idx] = context_input.subj_input
                obj_input_a[context_idx] = context_input.obj_input
                pobj_input_a[context_idx] = context_input.pobj_input

            eval_input_list_all = \
                rich_predicate.get_eval_input_list_all(include_salience=True)

            num_candidates = rich_predicate.num_candidates

            coherence_score_list_all = []

            for label, arg_idx, eval_input_list in eval_input_list_all:
                coherence_score_list = []

                arg_idx_input = \
                    np.asarray([float(arg_idx)] * num_context).astype(
                        np.float32)

                for eval_input, arg_salience in eval_input_list:
                    assert isinstance(eval_input, IndexedEvent), \
                        'eval_input must be a {} instance'.format(
                            get_class_name(IndexedEvent))
                    pred_input_b = np.asarray(
                        [eval_input.pred_input] * num_context).astype(np.int32)
                    subj_input_b = np.asarray(
                        [eval_input.subj_input] * num_context).astype(np.int32)
                    obj_input_b = np.asarray(
                        [eval_input.obj_input] * num_context).astype(np.int32)
                    pobj_input_b = np.asarray(
                        [eval_input.pobj_input] * num_context).astype(np.int32)

                    if use_salience:
                        if arg_salience is not None:
                            salience_feature = \
                                arg_salience.get_feature_list(salience_features)
                        else:
                            # NOBUG: this should never happen
                            print 'salience feature = None, filled with 0'
                            salience_feature = [0.0] * len(salience_features)

                        saliance_input = np.tile(
                            salience_feature, [num_context, 1]).astype(
                            np.float32)

                        coherence_output = coherence_fn(
                            pred_input_a, subj_input_a, obj_input_a,
                            pobj_input_a,
                            pred_input_b, subj_input_b, obj_input_b,
                            pobj_input_b,
                            arg_idx_input, saliance_input)
                    else:
                        coherence_output = coherence_fn(
                            pred_input_a, subj_input_a, obj_input_a,
                            pobj_input_a,
                            pred_input_b, subj_input_b, obj_input_b,
                            pobj_input_b,
                            arg_idx_input)

                    if use_max_score:
                        coherence_score_list.append(coherence_output.max())
                    else:
                        coherence_score_list.append(coherence_output.sum())

                assert len(coherence_score_list) == num_candidates
                coherence_score_list_all.append((label, coherence_score_list))

            num_label = len(eval_input_list_all)
            coherence_score_matrix = np.ndarray(
                shape=(num_label, num_candidates))
            row_idx = 0
            for label, coherence_score_list in coherence_score_list_all:
                coherence_score_matrix[row_idx, :] = np.array(
                    coherence_score_list)
                row_idx += 1

            for column_idx in range(num_candidates):
                max_coherence_score_idx = \
                    coherence_score_matrix[:, column_idx].argmax()
                for row_idx in range(num_label):
                    if row_idx != max_coherence_score_idx:
                        coherence_score_matrix[row_idx, column_idx] = -1.0

            for row_idx in range(num_label):
                assert coherence_score_list_all[row_idx][0] == \
                       rich_predicate.imp_args[row_idx].label
                rich_predicate.imp_args[row_idx].set_coherence_score_list(
                    coherence_score_matrix[row_idx, :])

        print 'Predicates with no context events:'
        for pred_idx in exclude_pred_idx_list:
            rich_predicate = self.all_rich_predicates[pred_idx]
            print 'Predicate #{}: {}, missing_imp_args = {}, imp_args = {}'.format(
                pred_idx,
                rich_predicate.n_pred,
                len(rich_predicate.imp_args),
                len([imp_arg for imp_arg in rich_predicate.imp_args
                     if imp_arg.exist]))

    def cross_val(self):
        assert len(self.all_rich_predicates) > 0

        optimized_thres = []

        for train, test in self.train_test_folds:
            thres_list = [float(x) / 100 for x in range(0, 100)]

            precision_list = []
            recall_list = []
            f1_list = []

            for thres in thres_list:
                total_dice = 0.0
                total_gt = 0.0
                total_model = 0.0

                for idx in train:
                    rich_predicate = self.all_rich_predicates[idx]
                    rich_predicate.eval(thres)

                    total_dice += rich_predicate.sum_dice
                    total_gt += rich_predicate.num_gt
                    total_model += rich_predicate.num_model

                precision, recall, f1 = \
                    consts.compute_f1(total_dice, total_gt, total_model)

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            max_f1 = max(f1_list)
            max_thres = thres_list[f1_list.index(max_f1)]
            optimized_thres.append(max_thres)

            for idx in test:
                rich_predicate = self.all_rich_predicates[idx]
                rich_predicate.thres = max_thres

        for rich_predicate in self.all_rich_predicates:
            rich_predicate.eval(rich_predicate.thres)

    def print_eval_stats(self):
        print_eval_stats(self.all_rich_predicates)

    def print_eval_results(self, use_corenlp_token=True):
        print_eval_results(self.all_rich_predicates,
                           use_corenlp_token=use_corenlp_token)
