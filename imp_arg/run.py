import sys
from os.path import exists

import consts
from event_comp_model import load_event_comp_model
from implicit_argument_reader import ImplicitArgumentReader


implicit_argument_reader = ImplicitArgumentReader()
implicit_argument_reader.read_dataset(sys.argv[1])

if exists(consts.all_predicates_path):
    implicit_argument_reader.load_predicates(consts.all_predicates_path)
else:
    implicit_argument_reader.build_predicates(consts.all_predicates_path)

implicit_argument_reader.add_candidates()

implicit_argument_reader.print_stats()

implicit_argument_reader.build_rich_predicates(use_corenlp_token=True)

event_comp_model = load_event_comp_model('40M_training_w_salience')
# event_comp_model = load_event_comp_model('8M_training_w_salience_w_wo_arg')
# event_comp_model = load_event_comp_model('8M_training_w_salience_w_two_args')

implicit_argument_reader.compute_coherence_score(
    event_comp_model, use_max_score=True)

implicit_argument_reader.cross_val(comp_wo_arg=True)

implicit_argument_reader.print_eval_stats()

# implicit_argument_reader.print_eval_results(use_corenlp_token=True)
