import pickle as pkl
import timeit

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from consts import *
from corpus_reader import CoreNLPReader
from event_comp_model import EventCompositionModel
from rich_predicate import RichPredicate
from rich_script.indexed_event import IndexedEvent
from util import consts, get_class_name
from stats import print_eval_stats


use_max_score = True


def cross_val(all_rich_predicates, n_splits):
    predicate_indices = range(len(all_rich_predicates))
    kf = KFold(n_splits=n_splits, shuffle=True)

    optimized_thres = []

    for train, test in kf.split(predicate_indices):
        train_idx_list = [predicate_indices[idx] for idx in train]

        thres_list = [float(x) / 100 for x in range(0, 100)]

        precision_list = []
        recall_list = []
        f1_list = []

        for thres in thres_list:
            total_dice = 0.0
            total_gt = 0.0
            total_model = 0.0

            for idx in train_idx_list:
                rich_predicate = all_rich_predicates[idx]
                rich_predicate.eval(thres)

                total_dice += rich_predicate.sum_dice
                total_gt += rich_predicate.num_gt
                total_model += rich_predicate.num_model

            precision, recall, f1 = \
                compute_f1(total_dice, total_gt, total_model)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        max_f1 = max(f1_list)
        max_thres = thres_list[f1_list.index(max_f1)]
        optimized_thres.append(max_thres)

        test_idx_list = [predicate_indices[idx] for idx in test]
        for idx in test_idx_list:
            rich_predicate = all_rich_predicates[idx]
            rich_predicate.thres = max_thres

    for rich_predicate in all_rich_predicates:
        rich_predicate.eval(rich_predicate.thres)


def compute_coherence_score():

    print '\nLoading all rich predicates from {}'.format(
        all_rich_predicates_path)
    start_time = timeit.default_timer()

    all_rich_predicates = pkl.load(open(all_rich_predicates_path, 'r'))
    assert all(isinstance(rich_predicate, RichPredicate)
               for rich_predicate in all_rich_predicates)

    elapsed = timeit.default_timer() - start_time
    print '\tDone in {:.3f} seconds'.format(elapsed)

    corenlp_reader = CoreNLPReader.load(corenlp_dict_path)

    event_comp_dir_dict = {
        '8M_training_w_salience':
            '/Users/pengxiang/corpora/spaces/20170519/fine_tuning_full/iter_13',
        '40M_training_w_salience':
            '/Users/pengxiang/corpora/spaces/20170530/fine_tuning_full/iter_19',
        '8M_training_wo_salience':
            '/Users/pengxiang/corpora/spaces/20170609/fine_tuning_full/iter_19',
        '40M_training_wo_salience':
            '/Users/pengxiang/corpora/spaces/20170611/fine_tuning_full/iter_19'
    }

    event_comp_dir = event_comp_dir_dict['40M_training_wo_salience']

    print '\nLoading event composition model from {}'.format(event_comp_dir)
    start_time = timeit.default_timer()

    event_comp_model = EventCompositionModel.load_model(event_comp_dir)
    word2vec_model = event_comp_model.word2vec

    coherence_fn = event_comp_model.pair_composition_network.coherence_fn
    use_salience = event_comp_model.pair_composition_network.use_salience

    elapsed = timeit.default_timer() - start_time
    print '\tDone in {:.3f} seconds'.format(elapsed)

    context_input_list_mapping = {}

    for rich_predicate in all_rich_predicates:
        if rich_predicate.fileid not in context_input_list_mapping:
            rich_script = corenlp_reader.get_rich_script(rich_predicate.fileid)
            rich_script.get_index(
                word2vec_model, include_type=True, use_unk=True)

            rich_event_list = rich_script.get_indexed_events()

            context_input_list = \
                [rich_event.get_pos_input(include_all_pobj=False)
                 for rich_event in rich_event_list]

            context_input_list_mapping[rich_predicate.fileid] = \
                context_input_list

    for rich_predicate in all_rich_predicates:
        rich_predicate.get_index(
            word2vec_model, include_type=True, use_unk=True)

    exclude_pred_idx_list = []

    pred_idx = 0
    for rich_predicate in tqdm(all_rich_predicates, desc='Processed', ncols=100):
        context_input_list = context_input_list_mapping[rich_predicate.fileid]
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
                np.asarray([float(arg_idx)] * num_context).astype(np.float32)

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
                        salience_feature = arg_salience.get_feature_list()
                    else:
                        salience_feature = [0.0] * consts.NUM_SALIENCE_FEATURES

                    saliance_input = np.tile(
                        salience_feature, [num_context, 1]).astype(np.float32)

                    coherence_output = coherence_fn(
                        pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
                        pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
                        arg_idx_input, saliance_input)
                else:
                    coherence_output = coherence_fn(
                        pred_input_a, subj_input_a, obj_input_a, pobj_input_a,
                        pred_input_b, subj_input_b, obj_input_b, pobj_input_b,
                        arg_idx_input)

                if use_max_score:
                    coherence_score_list.append(coherence_output.max())
                else:
                    coherence_score_list.append(coherence_output.sum())

            assert len(coherence_score_list) == num_candidates
            coherence_score_list_all.append((label, coherence_score_list))

        num_label = len(eval_input_list_all)
        coherence_score_matrix = np.ndarray(shape=(num_label, num_candidates))
        row_idx = 0
        for label, coherence_score_list in coherence_score_list_all:
            coherence_score_matrix[row_idx, :] = np.array(coherence_score_list)
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

    print '\nSaving all rich predicates with coherence scores to {}'.format(
        all_rich_predicates_with_coherence_path)
    pkl.dump(all_rich_predicates,
             open(all_rich_predicates_with_coherence_path, 'w'))

    for pred_idx in exclude_pred_idx_list:
        rich_predicate = all_rich_predicates[pred_idx]
        print 'Predicate #{}: {}, missing_imp_args = {}, imp_args = {}'.format(
            pred_idx,
            rich_predicate.n_pred,
            len(rich_predicate.imp_args),
            len([imp_arg for imp_arg in rich_predicate.imp_args
                 if imp_arg.exist]))


def evaluate():
    print '\nLoading all rich predicates with coherence scores from {}'.format(
        all_rich_predicates_with_coherence_path)
    start_time = timeit.default_timer()

    all_rich_predicates = \
        pkl.load(open(all_rich_predicates_with_coherence_path, 'r'))
    assert all(isinstance(rich_predicate, RichPredicate)
               for rich_predicate in all_rich_predicates)

    elapsed = timeit.default_timer() - start_time
    print '\tDone in {:.3f} seconds'.format(elapsed)

    cross_val(all_rich_predicates, n_splits=10)

    print_eval_stats(all_rich_predicates)


def main():
    compute_coherence_score()
    evaluate()


main()
