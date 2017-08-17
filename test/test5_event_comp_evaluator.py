import pickle as pkl

from evaluate import ArgumentCompositionEvaluator, EventCompositionEvaluator
from event_comp_model import load_event_comp_model

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

event_comp_model = \
    load_event_comp_model('8M_training_w_salience_only_num_mentions')

# event composition evaluator
evaluator = EventCompositionEvaluator(
    use_lemma=True,
    include_type=True,
    ignore_first_mention=False,
    filter_stop_events=True,
    use_max_score=True)

evaluator.set_model(event_comp_model)

evaluator.evaluate(all_scripts, ignore_first_mention=False)

evaluator.evaluate(all_scripts, ignore_first_mention=True)
'''

arg_comp_dir = \
    '/Users/pengxiang/corpora/spaces/20170521/pretraining'

arg_comp_model = EventCompositionModel.load_model(arg_comp_dir)

# argument composition evaluator
evaluator = ArgumentCompositionEvaluator(
    use_lemma=True,
    include_type=True,
    ignore_first_mention=False,
    filter_stop_events=True,
    use_max_score=True)

evaluator.set_model(arg_comp_model)

evaluator.evaluate(all_scripts, ignore_first_mention=False)

evaluator.evaluate(all_scripts, ignore_first_mention=True)
'''
