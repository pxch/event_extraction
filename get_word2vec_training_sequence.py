import argparse
from bz2 import BZ2File
from os import listdir
from os.path import isfile, join

from rich_script import RichScript, ScriptCorpus

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory for ScriptCorpus files')
parser.add_argument('output_path', help='path to write training sequence')
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

args = parser.parse_args()

fout = BZ2File(args.output_path, 'w')

input_files = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                      if isfile(join(args.input_path, f))
                      and f.endswith('.bz2')])

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
            sequence = rich_script.get_word2vec_training_seq()
            fout.write(' '.join(sequence) + '\n')

fout.close()
