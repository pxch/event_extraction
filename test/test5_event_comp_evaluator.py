import pickle as pkl

from evaluate import ArgumentCompositionEvaluator, EventCompositionEvaluator
from event_comp_model import EventCompositionModel
from os.path import join

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

root_dir = '/Users/pengxiang/corpora/spaces'
# root_dir = '/scratch/cluster/pxcheng/corpora/enwiki-20160901/event_comp_training/results'

event_comp_dir_dict = {
    '8M_training_w_salience':
        join(root_dir, '20170519/fine_tuning_full/iter_13'),
    '40M_training_w_salience':
        join(root_dir, '20170530/fine_tuning_full/iter_19'),
    '8M_training_wo_salience':
        join(root_dir, '20170609/fine_tuning_full/iter_19'),
    '40M_training_wo_salience':
        join(root_dir, '20170611/fine_tuning_full/iter_19'),
}

event_comp_dir = event_comp_dir_dict['40M_training_w_salience']

event_comp_model = EventCompositionModel.load_model(event_comp_dir)

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
