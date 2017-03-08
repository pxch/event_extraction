'''
import pickle as pkl

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

all_docs_dump_name = 'all_docs_03080300.pkl'
pkl.dump(all_docs, open(all_docs_dump_name, 'w'))

import event_script
all_scripts = []
for doc in all_docs:
    script = event_script.EventScript()
    script.read_from_document(doc)
    all_scripts.append(script)

all_scripts_dump_name = 'all_scripts_03080300.pkl'
pkl.dump(all_scripts, open(all_scripts_dump_name, 'w'))


'''
import pickle as pkl
all_scripts = pkl.load(open('all_scripts_03080300.pkl', 'r'))

import embedding
word2vec = embedding.Embedding('word2vec', False)
word2vec.load_model('/Users/pengxiang/corpora/ontonotes-release-5.0/output/dim300vecs.bin.gz', True)

from most_freq_coref_evaluator import MostFreqCorefEvaluator
most_freq_coref_eval = MostFreqCorefEvaluator()
most_freq_coref_eval.evaluate(all_scripts)

from most_sim_event_evaluator import MostSimEventEvaluator
most_sim_event_eval = MostSimEventEvaluator()
most_sim_event_eval.set_model(word2vec)
most_sim_event_eval.set_use_max_score(True)
most_sim_event_eval.evaluate(all_scripts)

