from evaluate.eval_stats import EvalStats


def eval_most_freq_coref(all_scripts):
    print 'Evaluation based on most frequent coreference chain:'
    eval_stats = EvalStats()
    for script in all_scripts:
        eval_stats.add_eval_stats(script.eval_most_freq_coref())
    print eval_stats.pretty_print()


def eval_most_sim_arg(all_scripts, model, use_other_args=False,
                      syntax_suffix=False, head_only=False, rep_only=False):
    print 'Evaluation based on most similar argument, ' + \
          'use_other_args = {}, head_only = {}, rep_only = {}:'.format(
              use_other_args, head_only, rep_only)
    eval_stats = EvalStats()
    for script in all_scripts:
        eval_stats.add_eval_stats(script.eval_most_sim_arg(
            model, use_other_args, syntax_suffix, head_only, rep_only))
    print eval_stats.pretty_print()


def eval_most_sim_event(all_scripts, model, use_max_score=False,
                        syntax_suffix=False, head_only=False, rep_only=False):
    print 'Evaluation based on most similar event, ' + \
          'use_max_score = {}, head_only = {}, rep_only = {}:'.format(
              use_max_score, head_only, rep_only)
    eval_stats = EvalStats()
    for script in all_scripts:
        eval_stats.add_eval_stats(script.eval_most_sim_event(
            model, use_max_score, syntax_suffix, head_only, rep_only))
    print eval_stats.pretty_print()


def eval_all(all_scripts, model, syntax_suffix, head_only, rep_only):
    eval_most_sim_arg(all_scripts, model, use_other_args=False,
                      syntax_suffix=syntax_suffix, head_only=head_only,
                      rep_only=rep_only)
    eval_most_sim_arg(all_scripts, model, use_other_args=True,
                      syntax_suffix=syntax_suffix, head_only=head_only,
                      rep_only=rep_only)

    eval_most_sim_event(all_scripts, model, use_max_score=False,
                        syntax_suffix=syntax_suffix, head_only=head_only,
                        rep_only=rep_only)
    eval_most_sim_event(all_scripts, model, use_max_score=True,
                        syntax_suffix=syntax_suffix, head_only=head_only,
                        rep_only=rep_only)