import abc
import logging

from eval_stats import EvalStats
from rich_script import RichScript, Script
from util import Word2VecModel, consts, get_class_name, read_vocab_list

logging.basicConfig(format='%(levelname) s : : %(message)s', level=logging.INFO)


class BaseEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, logger, use_lemma=True, include_type=True,
                 include_all_pobj=False, ignore_first_mention=False,
                 filter_stop_events=True):
        self.eval_stats = EvalStats()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.use_lemma = use_lemma
        self.include_type = include_type
        self.include_all_pobj = include_all_pobj
        self.ignore_first_mention = ignore_first_mention
        self.filter_stop_events = filter_stop_events
        self.model = None
        self.model_name = ''
        self.embedding_model = None
        self.embedding_model_name = ''

    @abc.abstractmethod
    def set_model(self, model):
        return

    def set_embedding_model(self, embedding_model):
        assert isinstance(embedding_model, Word2VecModel), \
            'model must be a {} instance'.format(get_class_name(Word2VecModel))
        self.logger.info('set embedding model: {}'.format(embedding_model.name))
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model.name

    def set_config(self, **kwargs):
        for key in kwargs:
            if key in self.__dict__:
                self.logger.debug('set {} = {}'.format(key, kwargs[key]))
                self.__dict__[key] = kwargs[key]
            else:
                self.logger.warning(
                    '{} is not a valid configuration keyword'.format(key))

    def log_evaluator_info(self):
        self.logger.info(
            'evaluation based on {}, with embedding model {}'.format(
                self.model_name, self.embedding_model_name))
        self.logger.info(
            'embedding configs: use_lemma = {}, include_type = {}'.format(
                self.use_lemma, self.include_type))
        self.logger.info(
            'general configs: include_all_pobj = {}, '
            'ignore_first_mention = {}, filter_stop_events = {}'.format(
                self.include_all_pobj, self.ignore_first_mention,
                self.filter_stop_events))

    # an argument should be ignored, if both self.ignore_first_mention is True,
    # and rich_argument.is_first_mention() returns True
    def ignore_argument(self, rich_argument):
        return self.ignore_first_mention and rich_argument.is_first_mention()

    @abc.abstractmethod
    def evaluate_event_list(self, rich_event_list):
        return

    def evaluate(self, all_scripts, **kwargs):
        self.set_config(**kwargs)
        self.log_evaluator_info()
        self.eval_stats = EvalStats()
        for script in all_scripts:
            assert isinstance(script, Script), \
                'every script in all_scripts must be a {} instance'.format(
                    get_class_name(Script))

            # ignore script where there is less than 2 events
            # (i.e., no context events to be compared to)
            if len(script.events) < 2:
                continue
            # ignore script where there is less than 2 entities
            # (i.e., only one candidate to select from)
            if len(script.entities) < 2:
                continue

            self.logger.debug('Processing script {}'.format(script.doc_name))

            # load prep_vocab_list
            prep_vocab_list = read_vocab_list(consts.PREP_VOCAB_LIST_FILE)

            # build the rich_script from script
            rich_script = RichScript.build(
                script,
                prep_vocab_list=prep_vocab_list,
                use_lemma=self.use_lemma,
                filter_stop_events=self.filter_stop_events
            )
            # index the rich_script with the embedding model
            rich_script.get_index(
                self.embedding_model,
                include_type=self.include_type,
                use_unk=True
            )

            # get the list of indexed events in the script
            rich_event_list = rich_script.get_indexed_events()
            # ignore rich_script where there is less than 2 indexed events
            # (i.e., no context events to be compared to)
            if len(rich_event_list) < 2:
                continue

            self.evaluate_event_list(rich_event_list)

        self.print_stats()

    def print_stats(self):
        print self.eval_stats.pretty_print()
