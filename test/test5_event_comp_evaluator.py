import pickle as pkl

from evaluate.event_comp_evaluator import EventCompositionEvaluator
from event_composition import EventCompositionModel

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

event_comp_dir = \
    '/Users/pengxiang/corpora/spaces/event_comp/iter-5'

event_comp_model = EventCompositionModel.load_from_directory(event_comp_dir)

evaluator = EventCompositionEvaluator(
    ignore_first_mention=False,
    include_prep=True,
    use_ner=True
)

evaluator.set_model(event_comp_model)
evaluator.evaluate(all_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_scripts)
