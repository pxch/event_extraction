import argparse
from bz2 import BZ2File
from collections import defaultdict, Counter
from os import listdir
from os.path import isdir, join

from util import read_counter, write_counter

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory to read vocabulary counts')
parser.add_argument('output_path', help='directory to write vocabulary counts')

args = parser.parse_args()

input_dirs = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                     if isdir(join(args.input_path, f))])

all_vocab = defaultdict(Counter)

for input_dir in input_dirs:
    print 'Reading vocabulary count from {}'.format(input_dir)
    with BZ2File(join(input_dir, 'argument.bz2'), 'r') as fin:
        all_vocab['argument'] += read_counter(fin)
    with BZ2File(join(input_dir, 'name_entity.bz2'), 'r') as fin:
        all_vocab['name_entity'] += read_counter(fin)
    with BZ2File(join(input_dir, 'name_entity_tag.bz2'), 'r') as fin:
        all_vocab['name_entity_tag'] += read_counter(fin)
    with BZ2File(join(input_dir, 'predicate.bz2'), 'r') as fin:
        all_vocab['predicate'] += read_counter(fin)
    with BZ2File(join(input_dir, 'preposition.bz2'), 'r') as fin:
        all_vocab['preposition'] += read_counter(fin)

for key in all_vocab:
    fout = BZ2File(join(args.output_path, key + '.bz2'), 'w')
    write_counter(all_vocab[key], fout)
    fout.close()
