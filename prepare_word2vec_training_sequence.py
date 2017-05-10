import argparse
from bz2 import BZ2File
from os import listdir
from os.path import isfile, join, dirname, realpath

from rich_script import RichScript, ScriptCorpus
from util import consts, read_vocab_list

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory to read ScriptCorpus files')
parser.add_argument('output_path', help='path to write training sequence')
parser.add_argument('--pred_vocab', help='path to predicate vocab file')
parser.add_argument('--arg_vocab', help='path to argument vocab file')
parser.add_argument('--ner_vocab', help='path to name entity vocab file')
parser.add_argument('--prep_vocab', help='path to preposition vocab file')

args = parser.parse_args()

fout = BZ2File(args.output_path, 'w')

input_files = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                      if isfile(join(args.input_path, f))
                      and f.endswith('.bz2')])

cur_dir_path = dirname(realpath(__file__))

if args.pred_vocab:
    pred_vocab_list = read_vocab_list(args.pred_vocab)
else:
    pred_vocab_list = read_vocab_list(
        join(cur_dir_path, consts.PRED_VOCAB_LIST_FILE))
if args.arg_vocab:
    arg_vocab_list = read_vocab_list(args.arg_vocab)
else:
    arg_vocab_list = read_vocab_list(
        join(cur_dir_path, consts.ARG_VOCAB_LIST_FILE))
if args.ner_vocab:
    ner_vocab_list = read_vocab_list(args.ner_vocab)
else:
    ner_vocab_list = read_vocab_list(
        join(cur_dir_path, consts.NER_VOCAB_LIST_FILE))
if args.prep_vocab:
    prep_vocab_list = read_vocab_list(args.prep_vocab)
else:
    prep_vocab_list = read_vocab_list(
        join(cur_dir_path, consts.PREP_VOCAB_LIST_FILE))

for input_f in input_files:
    with BZ2File(input_f, 'r') as fin:
        script_corpus = ScriptCorpus.from_text(fin.read())
        for script in script_corpus.scripts:
            print 'Reading script {}'.format(script.doc_name)
            rich_script = RichScript.build_with_vocab_list(
                script,
                pred_vocab_list=pred_vocab_list,
                arg_vocab_list=arg_vocab_list,
                ner_vocab_list=ner_vocab_list,
                prep_vocab_list=prep_vocab_list,
                use_entity=True
            )
            sequence = rich_script.get_word2vec_training_seq(
                include_type=True,
                include_all_pobj=True
            )
            if sequence:
                fout.write(' '.join(sequence) + '\n')

fout.close()
