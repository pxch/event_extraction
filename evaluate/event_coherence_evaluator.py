import abc

from base_evaluator import BaseEvaluator
from rich_script import RichScript, Script
from util import consts, get_class_name, read_vocab_list


class EventCoherenceEvaluator(BaseEvaluator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger=None, ignore_first_mention=False, use_lemma=True,
                 include_type=True, use_max_score=True, include_all_pobj=True,
                 filter_stop_events=True):
        super(EventCoherenceEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention
        )
        self.model_name = ''
        self.model = None
        self.embedding_model = None
        self.use_lemma = use_lemma
        self.include_type = include_type
        self.use_max_score = use_max_score
        self.include_all_pobj = include_all_pobj
        self.filter_stop_events = filter_stop_events

    @abc.abstractmethod
    def set_model(self, model):
        return

    def log_evaluator_info(self):
        self.logger.info(
            'Evaluation based on most coherent event, model = {}'.format(
                self.model_name))
        self.logger.info(
            'Embedding configs: use_lemma = {}, include_type = {}'.format(
                self.use_lemma, self.include_type))
        self.logger.info(
            'Evaluator configs: ignore_first_mention = {}, '
            'use_max_score = {}, include_all_pobj = {}, '
            'filter_stop_events = {}'.format(
                self.ignore_first_mention, self.use_max_score,
                self.include_all_pobj, self.filter_stop_events))

    @abc.abstractmethod
    def evaluate_event(self, eval_input_list_all, context_input_list):
        return

    def evaluate_script(self, script):
        assert isinstance(script, Script), \
            'evaluate_script must be called with a {} instance'.format(
                get_class_name(Script))
        self.logger.debug('Processing script {}'.format(script.doc_name))

        prep_vocab_list = read_vocab_list(consts.PREP_VOCAB_LIST_FILE)

        rich_script = RichScript.build(
            script,
            prep_vocab_list=prep_vocab_list,
            use_lemma=self.use_lemma,
            filter_stop_events=self.filter_stop_events
        )

        rich_script.get_index(self.embedding_model, self.include_type)
        # remove events with None embedding (pred_idx == -1)
        rich_event_list = [rich_event for rich_event in rich_script.rich_events
                           if rich_event.pred_idx != -1]
        pos_input_list = \
            [rich_event.get_pos_input(include_all_pobj=self.include_all_pobj)
                for rich_event in rich_event_list]
        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            context_input_list = \
                pos_input_list[:event_idx] + pos_input_list[event_idx + 1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=self.include_all_pobj)
            self.evaluate_event(eval_input_list_all, context_input_list)
