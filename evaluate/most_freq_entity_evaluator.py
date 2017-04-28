from base_evaluator import BaseEvaluator
import rich_script
from util import get_class_name


class MostFreqEntityEvaluator(BaseEvaluator):
    def __init__(self, logger=None, ignore_first_mention=False):
        super(MostFreqEntityEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention
        )

    @staticmethod
    def is_most_freq_entity(entity_idx, entity_freqs):
        entity_freqs[entity_idx] -= 1
        most_freq_entity_idx = entity_freqs.index(max(entity_freqs))
        entity_freqs[entity_idx] += 1
        return entity_idx == most_freq_entity_idx

    def log_evaluator_info(self):
        self.logger.info('Evaluation based on most frequent entity')
        self.logger.info('Evaluator configs: ignore_first_mention = {}'.format(
                self.ignore_first_mention))

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
