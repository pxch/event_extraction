import pickle as pkl
'''

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

all_docs_dump_name = 'all_docs_03200000.pkl'
pkl.dump(all_docs, open(all_docs_dump_name, 'w'))

import event_script
all_scripts = []
for doc in all_docs:
    script = event_script.EventScript()
    script.read_from_document(doc)
    all_scripts.append(script)

all_scripts_dump_name = 'all_scripts_03200000.pkl'
pkl.dump(all_scripts, open(all_scripts_dump_name, 'w'))

'''

all_scripts = pkl.load(open('all_scripts_03200000.pkl', 'r'))


from most_freq_coref_evaluator import MostFreqCorefEvaluator
most_freq_coref_eval = MostFreqCorefEvaluator()
most_freq_coref_eval.evaluate(all_scripts)

most_freq_coref_eval.set_ignore_first_mention(True)
most_freq_coref_eval.evaluate(all_scripts)

import embedding

word2vec = embedding.Embedding('word2vec', 300, syntax_label = False)
word2vec.load_model('/Users/pengxiang/corpora/spaces/enwiki-20160901/dim300vecs.bin.gz', True)

levy_deps = embedding.Embedding('levy_deps', 300, syntax_label = False)
levy_deps.load_model('/Users/pengxiang/corpora/spaces/levy_deps', False)

pair_triple = embedding.Embedding('event_based_pair_triple', 300, syntax_label = True)
pair_triple.load_model('/Users/pengxiang/corpora/spaces/enwiki-20160901/event_based/dim300vecs_w_surface_pair_c_lemma_triple', False)

event_model = embedding.Embedding('event_script', 300, syntax_label = True, use_ner = True, use_lemma = True, include_compounds = True)
event_model.load_model('/Users/pengxiang/corpora/spaces/03141230_dim300vecs.bin', True)

from most_sim_event_evaluator import MostSimEventEvaluator
most_sim_event_eval = MostSimEventEvaluator()
most_sim_event_eval.set_use_max_score(True)

most_sim_event_eval.set_model(word2vec)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(levy_deps)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(pair_triple)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(event_model)
most_sim_event_eval.evaluate(all_scripts)


most_sim_event_eval.set_ignore_first_mention(True)

most_sim_event_eval.set_model(word2vec)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(levy_deps)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(pair_triple)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(event_model)
most_sim_event_eval.evaluate(all_scripts)


most_sim_event_eval.set_ignore_first_mention(False)
most_sim_event_eval.set_rep_only(True)
most_sim_event_eval.set_head_only(True)

most_sim_event_eval.set_model(word2vec)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(levy_deps)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(pair_triple)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(event_model)
most_sim_event_eval.evaluate(all_scripts)


most_sim_event_eval.set_ignore_first_mention(True)

most_sim_event_eval.set_model(word2vec)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(levy_deps)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(pair_triple)
most_sim_event_eval.evaluate(all_scripts)

most_sim_event_eval.set_model(event_model)
most_sim_event_eval.evaluate(all_scripts)
