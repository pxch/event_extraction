import pickle as pkl
from collections import defaultdict, Counter
from os.path import join
import sys
from util import write_counter

all_scripts = pkl.load(open('all_scripts.pkl', 'r'))

all_vocab_count = defaultdict(Counter)

for script in all_scripts:
    vocab_count = script.get_vocab_count(use_lemma=True)
    for key in vocab_count:
        all_vocab_count[key] += vocab_count[key]


if len(sys.argv) > 1:
    output_path = sys.argv[1]
else:
    output_path = './tmp/'

for key in all_vocab_count:
    with open(join(output_path, key), 'w') as fout:
        write_counter(all_vocab_count[key], fout)
