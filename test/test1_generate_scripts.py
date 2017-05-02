import pickle as pkl
import sys

import event_script
import rich_script
from dataset import read_all_docs_from_ontonotes

date_tag = sys.argv[1]

all_docs = read_all_docs_from_ontonotes('english-bn-cnn')

all_docs_dump_name = 'all_docs_{}.pkl'.format(date_tag)
pkl.dump(all_docs, open(all_docs_dump_name, 'w'))

all_scripts = []
for doc in all_docs:
    script = event_script.EventScript.construct(doc)
    all_scripts.append(script)

all_scripts_dump_name = 'all_scripts_{}.pkl'.format(date_tag)
pkl.dump(all_scripts, open(all_scripts_dump_name, 'w'))

all_simple_scripts = []
for doc in all_docs:
    simple_script = rich_script.Script.from_doc(doc)
    all_simple_scripts.append(simple_script)

all_simple_scripts_dump_name = 'all_simple_scripts_{}.pkl'.format(date_tag)
pkl.dump(all_simple_scripts, open(all_simple_scripts_dump_name, 'w'))
