from ontonotes_reader import *

import pickle as pkl
import sys
from os.path import isdir, join
from os import makedirs

cfg_file = sys.argv[1]

cnn_all = load_ontonotes(cfg_file)

all_docs = []

script_dir = sys.argv[3]
if not isdir(script_dir):
    makedirs(script_dir)

'''
plain_text_dir = sys.argv[4]
if not isdir(plain_text_dir):
    makedirs(plain_text_dir)
'''

for subcorp in cnn_all:
    for coref_bank in subcorp['coref']:
        doc_id, doc = read_doc_from_ontonotes(coref_bank)
        all_docs.append((doc_id, doc))

        fout = open(join(script_dir, doc_id + '.script'), 'w')
        fout.write(doc.to_plain_text())
        fout.write(doc.script.pretty_print())
        fout.close()

pkl.dump(all_docs, open(sys.argv[2], 'w'))

'''
for subcorp in cnn_all:
    for doc in subcorp['document']:
        doc_id = doc.document_id.strip().split('@')[0].split('/')[-1]
        fout = open(join(plain_text_dir, doc_id + '.txt'), 'w')
        fout.write(doc.no_trace_text)
        fout.close()
'''
