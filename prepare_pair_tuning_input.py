import argparse
from bz2 import BZ2File
from os import listdir
from os.path import isfile, join

from rich_script import RichScript, ScriptCorpus
from util import Word2VecModel

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory for ScriptCorpus files')
parser.add_argument('output_path', help='path to write training sequence')
parser.add_argument('word2vec', help='path to word2vec vector file')
parser.add_argument('word2vec_vocab', help='path to word2vec vocab file')
parser.add_argument('--use_lemma', action='store_true',
                    help='if turned on, use the lemma form of a token,'
                         'otherwise use the word form')
parser.add_argument('--include_neg', action='store_true',
                    help='include negation for predicate when applicable')
parser.add_argument('--include_prt', action='store_true',
                    help='include particle for predicate when applicable')
parser.add_argument('--use_entity', action='store_true',
                    help='if turned on, use entity representation for arguments'
                         'with entity_idx, otherwise use token representation')
parser.add_argument('--use_ner', action='store_true',
                    help='if turned on, use ner tag for entities, otherwise use'
                         'head_token of rep_mention')
parser.add_argument('--include_prep', action='store_true',
                    help='include preposition word in pobj representations')
parser.add_argument('--neg_type', default='neg',
                    help='how to select negative samples, options: '
                         'one (one negative event and one left event), '
                         'neg (one left event for every negative event), '
                         'all (every left event for every negative event)')

args = parser.parse_args()

fout = BZ2File(args.output_path, 'w')

input_files = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                      if isfile(join(args.input_path, f))
                      and f.endswith('.bz2')])

model = Word2VecModel.load_model(
    args.word2vec, fvocab=args.word2vec_vocab, binary=True)


for input_f in input_files:
    with BZ2File(input_f, 'r') as fin:
        script_corpus = ScriptCorpus.from_text(fin.read())
        for script in script_corpus.scripts:
            rich_script = RichScript.build(
                script,
                use_lemma=args.use_lemma,
                include_neg=args.include_neg,
                include_prt=args.include_prt,
                use_entity=args.use_entity,
                use_ner=args.use_ner,
                include_prep=args.include_prep
            )
            rich_script.get_index(model)
            pair_tuning_inputs = rich_script.get_pair_tuning_input(
                neg_type=args.neg_type)
            if len(pair_tuning_inputs) > 0:
                fout.write('\n'.join(map(str, pair_tuning_inputs)) + '\n')

fout.close()
