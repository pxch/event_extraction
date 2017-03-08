import abc
from eval_stats import EvalStats


class BaseEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.eval_stats = EvalStats()

    @abc.abstractmethod
    def print_debug_message(self):
        return

    @abc.abstractmethod
    def evaluate_script(self, script):
        return

    def evaluate(self, all_scripts):
        print self.print_debug_message()
        self.eval_stats = EvalStats()
        for script in all_scripts:
            self.evaluate_script(script)
        self.print_stats()

    def print_stats(self):
        print self.eval_stats.pretty_print()
