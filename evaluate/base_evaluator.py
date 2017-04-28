import abc

from eval_stats import EvalStats
import logging

logging.basicConfig(format='%(levelname) s : : %(message)s', level=logging.INFO)


class BaseEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger, ignore_first_mention):
        self.eval_stats = EvalStats()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.ignore_first_mention = ignore_first_mention

    def set_ignore_first_mention(self, ignore_first_mention):
        self.logger.info(
            'Set ignore_first_mention = {}'.format(ignore_first_mention))
        self.ignore_first_mention = ignore_first_mention

    @abc.abstractmethod
    def log_evaluator_info(self):
        return

    @abc.abstractmethod
    def evaluate_script(self, script):
        return

    def evaluate(self, all_scripts):
        self.log_evaluator_info()
        self.eval_stats = EvalStats()
        for script in all_scripts:
            if len(script.events) < 2:
                continue
            if len(script.entities) < 2:
                continue
            self.evaluate_script(script)
        self.print_stats()

    def print_stats(self):
        print self.eval_stats.pretty_print()
