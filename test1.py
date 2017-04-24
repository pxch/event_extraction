import pickle as pkl

date_tag = '04232330'

ontonotes_source = '/Users/pengxiang/corpora/ontonotes-release-5.0/output/'

from ontonotes_loader import *

all_cnn = load_ontonotes(ontonotes_source + 'bn_cnn.conf')

all_coref_docs = get_all_coref_docs(all_cnn)
all_name_docs = get_all_name_docs(all_cnn)

import ontonotes_reader
all_docs = []
for coref_doc, name_doc in zip(all_coref_docs, all_name_docs):
    doc = ontonotes_reader.read_doc_from_ontonotes(coref_doc, name_doc)
    all_docs.append(doc)

all_docs_dump_name = 'all_docs_{}.pkl'.format(date_tag)
pkl.dump(all_docs, open(all_docs_dump_name, 'w'))

import event_script
all_scripts = []
for doc in all_docs:
    script = event_script.EventScript(doc.doc_name)
    script.read_from_document(doc)
    all_scripts.append(script)

all_scripts_dump_name = 'all_scripts_{}.pkl'.format(date_tag)
pkl.dump(all_scripts, open(all_scripts_dump_name, 'w'))

import rich_script
all_simple_scripts = []
for script in all_scripts:
    simple_script = rich_script.Script.from_script(script)
    all_simple_scripts.append(simple_script)

all_simple_scripts_dump_name = 'all_simple_scripts_{}.pkl'.format(date_tag)
pkl.dump(all_simple_scripts, open(all_simple_scripts_dump_name, 'w'))
