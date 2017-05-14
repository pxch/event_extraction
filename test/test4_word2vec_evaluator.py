import pickle as pkl

from evaluate.word2vec_evaluator import Word2VecEvaluator
from util import Word2VecModel

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

vector_file = \
    '/Users/pengxiang/corpora/spaces/event_scripts/' \
    'entity_1_ner_1_prep_1_dim300vecs.bin'
vocab_file = \
    '/Users/pengxiang/corpora/spaces/event_scripts/' \
    'entity_1_ner_1_prep_1_dim300vecs.vocab'

word2vec = Word2VecModel.load_model(vector_file, fvocab=vocab_file)

evaluator = Word2VecEvaluator(
    ignore_first_mention=False, include_all_pobj=False)

evaluator.set_model(word2vec)

evaluator.evaluate(all_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_scripts)
