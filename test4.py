import pickle as pkl

date_tag = '04232330'

all_scripts = pkl.load(open('all_scripts_{}.pkl'.format(date_tag), 'r'))

import embedding

event_model = embedding.Embedding('event_script', 300, syntax_label = True, use_ner = True, use_lemma = True, include_compounds = True)
event_model.load_model('/Users/pengxiang/corpora/spaces/03141230_dim300vecs.bin', True)

from evaluate import MostSimEventEvaluator

most_sim_event_eval = MostSimEventEvaluator()
most_sim_event_eval.set_use_max_score(True)

most_sim_event_eval.set_rep_only(True)
most_sim_event_eval.set_head_only(True)
most_sim_event_eval.set_ignore_first_mention(True)

most_sim_event_eval.set_model(event_model)
most_sim_event_eval.evaluate(all_scripts)
