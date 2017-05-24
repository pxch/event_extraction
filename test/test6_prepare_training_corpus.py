import pickle as pkl
from os import makedirs
from os.path import join, isdir

from rich_script import RichScript
from util import Word2VecModel
from util import consts, read_counter, read_vocab_list

pred_vocab_list = read_vocab_list(consts.PRED_VOCAB_LIST_FILE)
arg_vocab_list = read_vocab_list(consts.ARG_VOCAB_LIST_FILE)
ner_vocab_list = read_vocab_list(consts.NER_VOCAB_LIST_FILE)
prep_vocab_list = read_vocab_list(consts.PREP_VOCAB_LIST_FILE)

with open(consts.PRED_VOCAB_COUNT_FILE, 'r') as fin:
    pred_count_dict = read_counter(fin)

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

word2vec_dir = '/Users/pengxiang/corpora/spaces/20170521/sample_1e-4_min_500/'

vector_file = join(word2vec_dir, 'min_500_dim300vecs.bin')
vocab_file = join(word2vec_dir, 'min_500_dim300vecs.vocab')

word2vec = Word2VecModel.load_model(vector_file, fvocab=vocab_file)

output_dir = 'tmp/05142100'
if not isdir(output_dir):
    makedirs(output_dir)

fout_word2vec = open(join(output_dir, 'word2vec.txt'), 'w')
fout_pretraining = open(join(output_dir, 'pretraining.txt'), 'w')
fout_pair_tuning = open(join(output_dir, 'pair_tuning.txt'), 'w')

for script in all_scripts:
    rich_script = RichScript.build(
        script,
        prep_vocab_list=prep_vocab_list,
        use_lemma=True,
        filter_stop_events=False
    )
    rich_script.get_index(word2vec, include_type=True, use_unk=True,
                          pred_count_dict=pred_count_dict)

    word2vec_training_seq = rich_script.get_word2vec_training_seq(
        pred_vocab_list=pred_vocab_list,
        arg_vocab_list=arg_vocab_list,
        ner_vocab_list=ner_vocab_list,
        include_type=True,
        include_all_pobj=True
    )
    fout_word2vec.write(' '.join(word2vec_training_seq) + '\n')

    fout_pretraining.write('\n' + script.doc_name + '\n\n')
    pretraining_input_list = rich_script.get_pretraining_input_list()
    if pretraining_input_list:
        fout_pretraining.write(
            '\n'.join(map(str, pretraining_input_list)) + '\n')

    fout_pair_tuning.write('\n' + script.doc_name + '\n\n')
    pair_tuning_input_list = rich_script.get_pair_tuning_input_list(
        neg_sample_type='one')
    if pair_tuning_input_list:
        fout_pair_tuning.write(
            '\n'.join(map(str, pair_tuning_input_list)) + '\n')

fout_word2vec.close()
fout_pretraining.close()
fout_pair_tuning.close()
