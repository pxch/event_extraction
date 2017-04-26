import pickle as pkl

date_tag = '04232330'

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

'''
from evaluate import MostFreqEntityEvaluator

most_freq_entity_eval = MostFreqEntityEvaluator()
most_freq_entity_eval.evaluate(all_simple_scripts)

most_freq_entity_eval.set_ignore_first_mention(True)
most_freq_entity_eval.evaluate(all_simple_scripts)

'''

from util import Word2VecModel

word2vec = Word2VecModel.load_model(
    '/Users/pengxiang/corpora/spaces/03141230_dim300vecs.bin',
    fvocab='/Users/pengxiang/corpora/spaces/03141230_dim300vecs.vocab')

from evaluate.word2vec_evaluator import Word2VecEvaluator

evaluator = Word2VecEvaluator(include_neg=False)

evaluator.set_model(word2vec)
evaluator.set_ignore_first_mention(True)

evaluator.evaluate(all_simple_scripts)
