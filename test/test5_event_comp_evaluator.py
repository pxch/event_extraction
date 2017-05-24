import pickle as pkl

from evaluate import EventCompositionEvaluator
from event_comp_model import EventCompositionModel

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

event_comp_dir = '/Users/pengxiang/corpora/spaces/20170521/iter_13'

event_comp_model = EventCompositionModel.load_model(event_comp_dir)

evaluator = EventCompositionEvaluator(
    use_lemma=True,
    include_type=True,
    ignore_first_mention=False,
    filter_stop_events=True,
    use_max_score=True)

evaluator.set_model(event_comp_model)

evaluator.evaluate(all_scripts, use_max_score=False)

evaluator.evaluate(all_scripts, use_max_score=True)
