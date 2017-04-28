import pickle as pkl
import sys

from evaluate import MostFreqEntityEvaluator

date_tag = sys.argv[1]

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

most_freq_entity_eval = MostFreqEntityEvaluator()
most_freq_entity_eval.evaluate(all_simple_scripts)

most_freq_entity_eval.set_ignore_first_mention(True)
most_freq_entity_eval.evaluate(all_simple_scripts)
