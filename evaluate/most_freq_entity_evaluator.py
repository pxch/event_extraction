from base_evaluator import BaseEvaluator
import rich_script
from util import get_class_name


class MostFreqEntityEvaluator(BaseEvaluator):
    def __init__(self):
        BaseEvaluator.__init__(self)

    @staticmethod
    def is_most_freq_entity(entity_idx, entity_freqs):
        entity_freqs[entity_idx] -= 1
        most_freq_entity_idx = entity_freqs.index(max(entity_freqs))
        entity_freqs[entity_idx] += 1
        return entity_idx == most_freq_entity_idx

    def print_debug_message(self):
        return 'Evaluation based on most frequent coreference chain, ' \
               'ignore_first_mention = {}:'.format(self.ignore_first_mention)

    def evaluate_script(self, script):
        assert isinstance(script, rich_script.Script), \
            'evaluate_script must be called with a {} instance'.format(
                get_class_name(rich_script.Script))
        num_choices = len(script.entities)
        entity_freqs = [len(entity.mentions) for entity in script.entities]
        for event in script.events:
            for label, arg in event.get_all_args_with_entity(
                    include_arg_type=True):
                # do not evaluate the first mention of a coref chain,
                # as per the evaluation framework of implicit argument,
                # we must have other mentions in previous sentences
                if (not self.ignore_first_mention) \
                        or arg.mention_idx != 0:
                    self.eval_stats.add_eval_result(
                        label,
                        self.is_most_freq_entity(arg.entity_idx, entity_freqs),
                        num_choices)
        return self.eval_stats
