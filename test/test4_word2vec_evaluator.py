import pickle as pkl
from os.path import join

from evaluate import Word2VecEvaluator
from util import Word2VecModel

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

word2vec_dir = '/Users/pengxiang/corpora/spaces/20170521/sample_1e-4_min_500/'

vector_file = join(word2vec_dir, 'min_500_dim300vecs.bin')
vocab_file = join(word2vec_dir, 'min_500_dim300vecs.vocab')

word2vec = Word2VecModel.load_model(vector_file, fvocab=vocab_file)

evaluator = Word2VecEvaluator(
    use_lemma=True,
    include_type=True,
    include_all_pobj=False,
    ignore_first_mention=False,
    filter_stop_events=True,
    use_max_score=True)

evaluator.set_model(word2vec)

evaluator.evaluate(all_scripts, ignore_first_mention=False)

evaluator.evaluate(all_scripts, ignore_first_mention=True)
