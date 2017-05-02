import pickle as pkl

from evaluate import MostFreqEntityEvaluator

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

evaluator = MostFreqEntityEvaluator(ignore_first_mention=False)
evaluator.evaluate(all_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_scripts)
