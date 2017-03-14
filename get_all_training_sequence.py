from corenlp_reader import read_doc_from_corenlp
from event_script import EventScript
import sys
from os import listdir
from os.path import isdir, isfile, join
from bz2 import BZ2File

input_path = sys.argv[1]
output_path = sys.argv[2]

fout = BZ2File(output_path, 'w')

all_subdirs = sorted([join(input_path, subdir) for subdir in listdir(input_path) \
        if isdir(join(input_path, subdir))])

for subdir in all_subdirs:
    input_files = sorted([join(subdir, f) for f in listdir(subdir) \
            if isfile(join(subdir, f)) and f.endswith('xml.bz2')])
    for input_f in input_files:
        with BZ2File(input_f, 'r') as fin:
            doc = read_doc_from_corenlp(fin)
            script = EventScript()
            script.read_from_document(doc)
            training_seq = script.get_training_seq()
            if training_seq:
                fout.write(' '.join(script.get_training_seq()) + '\n')

fout.close()

