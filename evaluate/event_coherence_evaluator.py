import abc

from base_evaluator import BaseEvaluator
from rich_script import RichScript, Script
from util import get_class_name


class EventCoherenceEvaluator(BaseEvaluator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger=None, ignore_first_mention=False, use_lemma=True,
                 include_neg=True, include_prt=True, use_entity=True,
                 use_ner=True, include_prep=True, include_type=True,
                 use_max_score=True, include_all_pobj=True):
        super(EventCoherenceEvaluator, self).__init__(
            logger=logger,
            ignore_first_mention=ignore_first_mention
        )
        self.model_name = ''
        self.model = None
        self.embedding_model = None
        self.use_lemma = use_lemma
        self.include_neg = include_neg
        self.include_prt = include_prt
        self.use_entity = use_entity
        self.use_ner = use_ner
        self.include_prep = include_prep
        self.include_type = include_type
        self.use_max_score = use_max_score
        self.include_all_pobj = include_all_pobj

    @abc.abstractmethod
    def set_model(self, model):
        return

    def log_evaluator_info(self):
        self.logger.info(
            'Evaluation based on most coherent event, model = {}'.format(
                self.model_name))
        self.logger.info(
            'Embedding configs: use_lemma = {}, include_neg = {}, '
            'include_prt = {}, use_entity = {}, use_ner = {}, '
            'include_prep = {}, include_type = {}'.format(
                self.use_lemma, self.include_neg, self.include_prt,
                self.use_entity, self.use_ner, self.include_prep,
                self.include_type))
        self.logger.info(
            'Evaluator configs: ignore_first_mention = {}, '
            'use_max_score = {}, include_all_pobj = {}'.format(
                self.ignore_first_mention, self.use_max_score,
                self.include_all_pobj))

    @abc.abstractmethod
    def evaluate_event(self, eval_input_list_all, context_input_list):
        return

    def evaluate_script(self, script):
        assert isinstance(script, Script), \
            'evaluate_script must be called with a {} instance'.format(
                get_class_name(Script))
        self.logger.debug('Processing script {}'.format(script.doc_name))

        rich_script = RichScript.build(
            script,
            use_lemma=self.use_lemma,
            include_neg=self.include_neg,
            include_prt=self.include_prt,
            use_entity=self.use_entity,
            use_ner=self.use_ner,
            include_prep=self.include_prep
        )
        rich_script.get_index(self.embedding_model, self.include_type)
        # remove events with None embedding (pred_idx == -1)
        rich_event_list = [rich_event for rich_event in rich_script.rich_events
                           if rich_event.pred_idx != -1]
        pos_input_list = \
            [rich_event.get_pos_training_input(
                include_all_pobj=self.include_all_pobj)
                for rich_event in rich_event_list]
        for event_idx, rich_event in enumerate(rich_event_list):
            self.logger.debug('Processing event #{}'.format(event_idx))
            context_input_list = \
                pos_input_list[:event_idx] + pos_input_list[event_idx + 1:]

            eval_input_list_all = rich_event.get_eval_input_list_all(
                include_all_pobj=self.include_all_pobj)
            self.evaluate_event(eval_input_list_all, context_input_list)
