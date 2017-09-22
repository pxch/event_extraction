import argparse
from os.path import exists

import consts
from implicit_argument_reader import ImplicitArgumentReader
from util import Word2VecModel

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='path to the implicit argument dataset')
parser.add_argument('output_path', help='directory to write training sequence')
parser.add_argument('word2vec', help='path to word2vec vector file')
parser.add_argument('word2vec_vocab', help='path to word2vec vocab file')

args = parser.parse_args()

implicit_argument_reader = ImplicitArgumentReader()
implicit_argument_reader.read_dataset(args.input_path)

if exists(consts.all_predicates_path):
    implicit_argument_reader.load_predicates(consts.all_predicates_path)
else:
    implicit_argument_reader.build_predicates(consts.all_predicates_path)

implicit_argument_reader.add_candidates()

implicit_argument_reader.print_stats()

implicit_argument_reader.build_rich_predicates(
    use_corenlp_token=True, labeled_arg_only=False)

word2vec_model = Word2VecModel.load_model(
    args.word2vec, fvocab=args.word2vec_vocab, binary=True)

output_dir = args.output_path

implicit_argument_reader.generate_cross_val_training_dataset(
    word2vec_model, output_dir)
