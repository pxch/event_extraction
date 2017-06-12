import os
import pickle as pkl

from corenlp import read_doc_from_corenlp
from ontonotes import read_all_docs_from_ontonotes
from rich_script import Script

all_docs = read_all_docs_from_ontonotes('english-bn-cnn')

pkl.dump(all_docs, open('all_docs.pkl', 'w'))

all_scripts = []
for doc in all_docs:
    script = Script.from_doc(doc)
    all_scripts.append(script)

pkl.dump(all_scripts, open('all_scripts.pkl', 'w'))

corenlp_path = '/Users/pengxiang/corpora/ontonotes-release-5.0/output/parsed'

corenlp_files = sorted([os.path.join(corenlp_path, f) for f
                        in os.listdir(corenlp_path) if f.endswith('.xml')])

all_docs_corenlp = [
    read_doc_from_corenlp(open(corenlp_file))
    for corenlp_file in corenlp_files]

pkl.dump(all_docs_corenlp, open('all_docs_corenlp.pkl', 'w'))

all_scripts_corenlp = []
for doc in all_docs_corenlp:
    script = Script.from_doc(doc)
    all_scripts_corenlp.append(script)

pkl.dump(all_scripts_corenlp, open('all_scripts_corenlp.pkl', 'w'))
