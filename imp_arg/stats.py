from collections import defaultdict
from operator import itemgetter

from texttable import Texttable

from consts import compute_f1


def print_stats(all_predicates):
    predicates_by_pred = defaultdict(list)

    for predicate in all_predicates:
        predicates_by_pred[predicate.n_pred].append(predicate)

    num_dict = {}

    for n_pred, predicates in predicates_by_pred.items():
        num_dict[n_pred] = [len(predicates)]
        num_dict[n_pred].append(
            sum([predicate.num_imp_arg() for predicate in predicates]))
        num_dict[n_pred].append(
            sum([predicate.num_imp_arg(2) for predicate in predicates]))
        num_dict[n_pred].append(
            sum([predicate.num_oracle() for predicate in predicates]))
        num_dict[n_pred].append(
            sum([1 for predicate in predicates if
                 'arg0' in predicate.imp_args]))
        num_dict[n_pred].append(
            sum([1 for predicate in predicates if
                 'arg1' in predicate.imp_args]))
        num_dict[n_pred].append(
            sum([1 for predicate in predicates if
                 'arg2' in predicate.imp_args]))
        num_dict[n_pred].append(
            sum([1 for predicate in predicates if
                 'arg3' in predicate.imp_args]))
        num_dict[n_pred].append(
            sum([1 for predicate in predicates if
                 'arg4' in predicate.imp_args]))

    total_pred = 0
    total_arg = 0
    total_arg_in_range = 0
    total_oracle_arg = 0
    total_imp_arg0 = 0
    total_imp_arg1 = 0
    total_imp_arg2 = 0
    total_imp_arg3 = 0
    total_imp_arg4 = 0

    table_content = []

    for n_pred, num in num_dict.items():
        table_row = [n_pred] + num[:2]
        table_row.append(float(num[1]) / num[0])
        table_row.append(num[2])
        table_row.append(num[3])
        table_row.append(100. * float(num[3]) / num[1])
        table_row += num[4:]
        table_content.append(table_row)

        total_pred += num[0]
        total_arg += num[1]
        total_arg_in_range += num[2]
        total_oracle_arg += num[3]
        total_imp_arg0 += num[4]
        total_imp_arg1 += num[5]
        total_imp_arg2 += num[6]
        total_imp_arg3 += num[7]
        total_imp_arg4 += num[8]

    table_content.sort(key=itemgetter(2), reverse=True)
    table_content.append([''] * 12)
    table_content.append([
        'Overall',
        total_pred,
        total_arg,
        float(total_arg) / total_pred,
        total_arg_in_range,
        total_oracle_arg,
        100. * float(total_oracle_arg) / total_arg,
        total_imp_arg0,
        total_imp_arg1,
        total_imp_arg2,
        total_imp_arg3,
        total_imp_arg4
    ])

    table_header = [
        'Pred.', '# Pred.', '# Imp.Arg.', '# Imp./pred.', '# Imp.Arg.in.range',
        '# Oracle', 'Oracle Recall', '# Imp.Arg.0', '# Imp.Arg.1',
        '# Imp.Arg.2', '# Imp.Arg.3', '# Imp.Arg.4']

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_cols_align(['c'] * len(table_header))
    table.set_cols_valign(['m'] * len(table_header))
    table.set_cols_width([15] * len(table_header))
    table.set_precision(2)

    table.header(table_header)
    for row in table_content:
        table.add_row(row)

    print table.draw()


def print_eval_stats(all_rich_predicates):
    predicates_by_pred = defaultdict(list)

    for rich_predicate in all_rich_predicates:
        predicates_by_pred[rich_predicate.n_pred].append(rich_predicate)

    num_dict = {}

    total_dice = 0.0
    total_gt = 0.0
    total_model = 0.0

    for n_pred, predicates in predicates_by_pred.items():
        num_dict[n_pred] = [len(predicates)]
        num_dict[n_pred].append(
            sum([predicate.num_imp_args() for predicate in predicates]))

        pred_dice = 0.0
        pred_gt = 0.0
        pred_model = 0.0
        for predicate in predicates:
            pred_dice += predicate.sum_dice
            pred_gt += predicate.num_gt
            pred_model += predicate.num_model

        total_dice += pred_dice
        total_gt += pred_gt
        total_model += pred_model

        precision, recall, f1 = compute_f1(pred_dice, pred_gt, pred_model)

        num_dict[n_pred].append(precision * 100)
        num_dict[n_pred].append(recall * 100)
        num_dict[n_pred].append(f1 * 100)

    total_precision, total_recall, total_f1 = \
        compute_f1(total_dice, total_gt, total_model)

    total_pred = 0
    total_arg = 0

    table_content = []

    for n_pred, num in num_dict.items():
        table_row = [n_pred] + num
        table_content.append(table_row)

        total_pred += num[0]
        total_arg += num[1]

    table_content.sort(key=itemgetter(2), reverse=True)
    table_content.append([''] * 6)
    table_content.append([
        'Overall',
        total_pred,
        total_arg,
        total_precision * 100,
        total_recall * 100,
        total_f1 * 100])

    table_header = [
        'Pred.', '# Pred.', '# Imp.Arg.', 'Precision', 'Recall', 'F1']

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_precision(2)

    table.header(table_header)
    for row in table_content:
        table.add_row(row)

    print table.draw()


def print_eval_results(all_rich_predicates):
    for predicate in all_rich_predicates:
        print
        print '{} {} {}\tthreshold = {}'.format(
            predicate.fileid, predicate.n_pred, predicate.v_pred,
            predicate.thres)
        for exp_arg in predicate.exp_args:
            print '{} : {}'.format(exp_arg.arg_type, exp_arg.core)
        if predicate.imp_args:
            header = ['Candidate', 'Surface', 'Core']
            for imp_arg in predicate.imp_args:
                arg_label = imp_arg.label
                if not imp_arg.exist:
                    arg_label = '(' + arg_label + ')'
                header.append(arg_label + ' dice')
                header.append(arg_label + ' coh')

            table = Texttable()
            table.set_deco(Texttable.BORDER | Texttable.HEADER |
                           Texttable.HLINES)
            table.set_cols_align(['c'] * len(header))
            table.set_cols_valign(['m'] * len(header))
            table.set_cols_width([25, 25, 25] + [12] * (len(header) - 3))
            table.set_cols_dtype(['t'] * len(header))
            table.set_precision(2)

            table.header(header)

            content = [[] for _ in range(predicate.num_candidates)]

            if not predicate.imp_args[0].has_coherence_score:
                continue

            for arg_idx, imp_arg in enumerate(predicate.imp_args):
                for candidate_idx, candidate in enumerate(
                        imp_arg.rich_candidate_list):
                    if arg_idx == 0:
                        content[candidate_idx].append(
                            '{} #{}'.format(candidate.arg_pointer,
                                            candidate.arg_pointer.entity_idx))
                        content[candidate_idx].append(
                            candidate.arg_pointer.corenlp_lemma_surface)
                        content[candidate_idx].append(str(candidate.core))
                    content[candidate_idx].append(
                        '{0:.2f}'.format(candidate.dice_score))
                    try:
                        coherence_score = \
                            imp_arg.coherence_score_list[candidate_idx]
                    except IndexError:
                        print 'arg_idx = {}, has_coherence_score = {}'.format(
                            arg_idx, imp_arg.has_coherence_score)
                        exit(-1)
                    coherence_str = '{0:.2f}'.format(coherence_score)
                    if candidate_idx == imp_arg.max_coherence_score_idx:
                        if coherence_score >= predicate.thres:
                            coherence_str += ' ***'
                        else:
                            coherence_str += ' *'
                    content[candidate_idx].append(coherence_str)

            for row in content:
                table.add_row(row)

            print table.draw()
            print
