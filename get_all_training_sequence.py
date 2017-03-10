from corenlp_reader import read_doc_from_corenlp
from event_script import EventScript
import sys
from os import listdir
from os.path import isdir, isfile, join

input_dir = sys.argv[1]
output_path = sys.argv[2]

fout = open(output_path, 'w')

for file_name in listdir(input_dir):
    file_path = join(input_dir, file_name)
    if not isfile(file_path):
        continue
    # TODO: switch to other file handlers when compressed
    fin = open(file_path, 'r')
    doc = read_doc_from_corenlp(fin)
    script = EventScript()
    script.read_from_document(doc)
    fout.write(' '.join(script.get_training_seq()) + '\n')

fout.close()

