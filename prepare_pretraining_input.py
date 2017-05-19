import argparse
from bz2 import BZ2File
from os import listdir
from os.path import isfile, join, dirname, realpath

from rich_script import RichScript, ScriptCorpus
from util import Word2VecModel, consts, read_counter, read_vocab_list

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory for ScriptCorpus files')
parser.add_argument('output_path', help='path to write training sequence')
parser.add_argument('word2vec', help='path to word2vec vector file')
parser.add_argument('word2vec_vocab', help='path to word2vec vocab file')
parser.add_argument('--prep_vocab', help='path to preposition vocab file')
parser.add_argument('--use_lemma', action='store_true',
                    help='if turned on, use the lemma form of a token,'
                         'otherwise use the word form')
parser.add_argument('--subsampling', action='store_true',
                    help='if turned on, most frequent predicates would be '
                         'randomly subsampled according to their frequency')

args = parser.parse_args()

fout = BZ2File(args.output_path, 'w')

input_files = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                      if isfile(join(args.input_path, f))
                      and f.endswith('.bz2')])

model = Word2VecModel.load_model(
    args.word2vec, fvocab=args.word2vec_vocab, binary=True)

cur_dir_path = dirname(realpath(__file__))

if args.prep_vocab:
    prep_vocab_list = read_vocab_list(args.prep_vocab)
else:
    prep_vocab_list = read_vocab_list(
        join(cur_dir_path, consts.PREP_VOCAB_LIST_FILE))

pred_count_dict = None
if args.subsampling:
    with open(join(cur_dir_path, consts.PRED_VOCAB_COUNT_FILE)) as fin:
        pred_count_dict = read_counter(fin)

for input_f in input_files:
    with BZ2File(input_f, 'r') as fin:
        script_corpus = ScriptCorpus.from_text(fin.read())
        for script in script_corpus.scripts:
            rich_script = RichScript.build(
                script,
                prep_vocab_list=prep_vocab_list,
                use_lemma=args.use_lemma,
                filter_stop_events=False
            )
            rich_script.get_index(model, include_type=True, use_unk=True,
                                  pred_count_dict=pred_count_dict)
            pretraining_inputs = rich_script.get_pretraining_input_list()
            if len(pretraining_inputs) > 0:
                fout.write('\n'.join(map(str, pretraining_inputs)) + '\n')

fout.close()
