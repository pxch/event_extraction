import pickle as pkl
import sys

from evaluate.word2vec_evaluator import Word2VecEvaluator
from util import Word2VecModel

date_tag = sys.argv[1]

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

vector_file = \
    '/Users/pengxiang/corpora/spaces/event_scripts/' \
    'entity_1_ner_1_prep_1_dim300vecs.bin'
vocab_file = \
    '/Users/pengxiang/corpora/spaces/event_scripts/' \
    'entity_1_ner_1_prep_1_dim300vecs.vocab'

word2vec = Word2VecModel.load_model(vector_file, fvocab=vocab_file)

evaluator = Word2VecEvaluator(
    ignore_first_mention=False,
    include_prep=True,
    use_ner=True,
    include_all_pobj=False
)

evaluator.set_model(word2vec)

evaluator.evaluate(all_simple_scripts)

evaluator.set_ignore_first_mention(True)
evaluator.evaluate(all_simple_scripts)
