import pickle as pkl

from ontonotes import read_all_docs_from_ontonotes
from rich_script import Script

all_docs = read_all_docs_from_ontonotes('english-bn-cnn')

pkl.dump(all_docs, open('all_docs.pkl', 'w'))

all_scripts = []
for doc in all_docs:
    script = Script.from_doc(doc)
    all_scripts.append(script)

pkl.dump(all_scripts, open('all_scripts.pkl', 'w'))
