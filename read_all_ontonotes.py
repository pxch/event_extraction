from ontonotes_reader import *

import pickle as pkl
import sys
from os.path import isdir, join
from os import makedirs

ontonotes_source = '/Users/pengxiang/corpora/ontonotes-release-5.0/data/files/data/english/annotations/'

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
        doc_id = coref_bank.document_id.split('@')[0]
        print 'Processing document {}'.format(doc_id)

        conll_file_path = join(ontonotes_source, doc_id + '.depparse')

        all_sents = read_conll_depparse(conll_file_path)
        all_corefs = read_coref_bank(coref_bank)

        doc = Doc.construct(all_sents, all_corefs)

        doc.fix_ontonotes_coref_info()

        doc.preprocessing()
        doc.extract_event_script()

        doc_id = doc_id.split('/')[-1]
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
