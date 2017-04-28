import pickle as pkl
import sys

from evaluate.event_comp_evaluator import EventCompositionEvaluator
from event_composition import EventCompositionModel

date_tag = sys.argv[1]

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

event_comp_dir = \
    '/Users/pengxiang/corpora/spaces/event_comp/iter-5'

event_comp_model = EventCompositionModel.load_from_directory(event_comp_dir)

evaluator = EventCompositionEvaluator(
    ignore_first_mention=False,
    include_prep=True,
    use_ner=True
)

evaluator.set_model(event_comp_model)
evaluator.evaluate(all_simple_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_simple_scripts)
