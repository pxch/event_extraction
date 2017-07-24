from collections import defaultdict
from operator import itemgetter

from prettytable import PrettyTable
from consts import compute_f1


def print_table(head, content):
    table = PrettyTable(head)
    for row in content:
        table.add_row(row)
    print table


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
        table_row.append('{0:.1f}'.format(float(num[1]) / num[0]))
        table_row.append(num[2])
        table_row.append(num[3])
        table_row.append('{0:.1f}'.format(100. * float(num[3]) / num[1]))
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
        '{0:.1f}'.format(float(total_arg) / total_pred),
        total_arg_in_range,
        total_oracle_arg,
        '{0:.1f}'.format(100. * float(total_oracle_arg) / total_arg),
        total_imp_arg0,
        total_imp_arg1,
        total_imp_arg2,
        total_imp_arg3,
        total_imp_arg4
    ])

    table_head = [
        'Pred.', '# Pred.', '# Imp.Arg.', '# Imp./pred.', '# Imp.Arg.in.range',
        '# Oracle', 'Oracle Recall', '# Imp.Arg.0', '# Imp.Arg.1',
        '# Imp.Arg.2', '# Imp.Arg.3', '# Imp.Arg.4']
    print_table(table_head, table_content)


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

        num_dict[n_pred].append('{0:.2f}'.format(precision * 100))
        num_dict[n_pred].append('{0:.2f}'.format(recall * 100))
        num_dict[n_pred].append('{0:.2f}'.format(f1 * 100))

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
        '{0:.2f}'.format(total_precision * 100),
        '{0:.2f}'.format(total_recall * 100),
        '{0:.2f}'.format(total_f1 * 100)])

    table_head = [
        'Pred.', '# Pred.', '# Imp.Arg.', 'Precision', 'Recall', 'F1']
    print_table(table_head, table_content)
