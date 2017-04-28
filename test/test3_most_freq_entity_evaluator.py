import pickle as pkl
import sys

from evaluate import MostFreqEntityEvaluator

date_tag = sys.argv[1]

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

evaluator = MostFreqEntityEvaluator(ignore_first_mention=False)
evaluator.evaluate(all_simple_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_simple_scripts)
