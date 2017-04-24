import pickle as pkl

date_tag = '04232330'

all_scripts = pkl.load(open('all_scripts_{}.pkl'.format(date_tag), 'r'))

from evaluate import MostFreqCorefEvaluator

most_freq_coref_eval = MostFreqCorefEvaluator()
most_freq_coref_eval.evaluate(all_scripts)

most_freq_coref_eval.set_ignore_first_mention(True)
most_freq_coref_eval.evaluate(all_scripts)

all_simple_scripts = pkl.load(
    open('all_simple_scripts_{}.pkl'.format(date_tag), 'r'))

from evaluate import MostFreqEntityEvaluator

most_freq_entity_eval = MostFreqEntityEvaluator()
most_freq_entity_eval.evaluate(all_simple_scripts)

most_freq_entity_eval.set_ignore_first_mention(True)
most_freq_entity_eval.evaluate(all_simple_scripts)

import embedding

word2vec = embedding.Embedding('word2vec', 300, syntax_label = False)
word2vec.load_model('/Users/pengxiang/corpora/spaces/enwiki-20160901/dim300vecs.bin.gz', True)

levy_deps = embedding.Embedding('levy_deps', 300, syntax_label = False)
levy_deps.load_model('/Users/pengxiang/corpora/spaces/levy_deps', False)

pair_triple = embedding.Embedding('event_based_pair_triple', 300, syntax_label = True)
pair_triple.load_model('/Users/pengxiang/corpora/spaces/enwiki-20160901/event_based/dim300vecs_w_surface_pair_c_lemma_triple', False)

event_model = embedding.Embedding('event_script', 300, syntax_label = True, use_ner = True, use_lemma = True, include_compounds = True)
event_model.load_model('/Users/pengxiang/corpora/spaces/03141230_dim300vecs.bin', True)

from evaluate import MostSimEventEvaluator

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
