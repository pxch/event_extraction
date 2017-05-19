import os

from autoencoder import DenoisingAutoencoderIterableTrainer
from event_composition_model import EventCompositionModel
from pair_composition_trainer import PairCompositionTrainer
from rich_script import PretrainingCorpusIterator
from util import get_class_name, get_console_logger


class EventCompositionTrainer(object):
    def __init__(self, model, tmp_dir, log=None):
        assert isinstance(model, EventCompositionModel), \
            'model must be a {} instance'.format(
                get_class_name(EventCompositionModel))
        self.model = model
        # temporary directory to save intermediate models
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.tmp_dir = tmp_dir
        if log is None:
            self.log = get_console_logger('event_comp_trainer')
        else:
            self.log = log

    def save_model(self, saving_dir, save_word2vec=True,
                   save_event_vector=True, save_pair_composition=False):
        saving_dir = os.path.join(self.tmp_dir, saving_dir)
        self.log.info('Saving model to {}'.format(saving_dir))
        self.model.save_model(
            saving_dir,
            save_word2vec=save_word2vec,
            save_event_vector=save_event_vector,
            save_pair_composition=save_pair_composition
        )

    def autoencoder_pretraining(
            self, indexed_corpus, batch_size=1000, iterations=2,
            learning_rate=0.1, regularization=0.001, corruption_level=0.3):
        self.log.info('Start autoencoder pre-training')
        self.log.info(
            'Pre-training with l2 reg={}, lr={}, corruption={}, '
            '{} iterations per layer, {}-instance minibatches'.format(
                regularization, learning_rate, corruption_level, iterations,
                batch_size))

        if not os.path.isdir(indexed_corpus):
            self.log.error(
                'Cannot find indexed corpus at {}'.format(indexed_corpus))
            exit(-1)

        self.save_model(
            os.path.join('pretraining', 'init'),
            save_word2vec=False,
            save_event_vector=True,
            save_pair_composition=False
        )

        for layer in range(len(self.model.event_vector_network.layer_sizes)):
            self.log.info('Pre-training layer {}'.format(layer))

            self.log.info(
                'Loading indexed corpus from: {}, with batch_size={}'.format(
                    indexed_corpus, batch_size))
            corpus_it = PretrainingCorpusIterator(
                indexed_corpus,
                model=self.model.event_vector_network,
                layer_input=layer,
                batch_size=batch_size)
            self.log.info('Found {} lines in the corpus'.format(len(corpus_it)))

            trainer = DenoisingAutoencoderIterableTrainer(
                self.model.event_vector_network.layers[layer])
            trainer.train(
                batch_iterator=corpus_it,
                iterations=iterations,
                log=self.log,
                learning_rate=learning_rate,
                regularization=regularization,
                corruption_level=corruption_level,
                loss='l2'
            )

            self.log.info('Finished training layer {}'.format(layer))
            self.save_model(
                os.path.join('pretraining', 'layer_{}'.format(layer)),
                save_word2vec=False,
                save_event_vector=True,
                save_pair_composition=False
            )

        self.log.info('Finished autoencoder pretraining')
        self.save_model(
            os.path.join('pretraining', 'finish'),
            save_word2vec=True,
            save_event_vector=True,
            save_pair_composition=False
        )

    def fine_tuning(
            self, batch_iterator, iterations=3, learning_rate=0.025,
            min_learning_rate=0.001, regularization=0.01,
            update_event_vectors=False, update_input_vectors=False,
            update_empty_vectors=False):
        self.log.info('Started pair composition fine tuning')
        self.log.info(
            'Fine tuning with l2 reg={}, lr={}, min_lr={}, {} iterations, '
            '{}-instance minibatches, {}updating event vectors, '
            '{}updating input vectors, {}updating empty vectors'.format(
                regularization, learning_rate, min_learning_rate,
                iterations, batch_iterator.batch_size,
                '' if update_event_vectors else 'not ',
                '' if update_input_vectors else 'not ',
                '' if update_empty_vectors else 'not '))

        saving_dir = 'fine_tuning'
        if update_event_vectors or update_input_vectors or update_empty_vectors:
            saving_dir = 'fine_tuning_full'

        save_word2vec = False
        if update_input_vectors:
            save_word2vec = True

        save_event_vector = False
        if update_event_vectors or update_empty_vectors:
            save_event_vector = True

        def _iteration_callback(iter_num):
            self.save_model(
                os.path.join(saving_dir, 'iter_{}'.format(iter_num)),
                save_word2vec=save_word2vec,
                save_event_vector=save_event_vector,
                save_pair_composition=True
            )

        self.save_model(
            os.path.join(saving_dir, 'init'),
            save_word2vec=save_word2vec,
            save_event_vector=save_event_vector,
            save_pair_composition=True
        )

        trainer = PairCompositionTrainer(
            self.model.pair_composition_network,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            regularization=regularization,
            update_event_vectors=update_event_vectors,
            update_input_vectors=update_input_vectors,
            update_empty_vectors=update_empty_vectors
        )
        trainer.train(
            batch_iterator,
            iterations=iterations,
            # TODO: add iteration_callback
            iteration_callback=_iteration_callback,
            log=self.log
        )

        self.save_model(
            os.path.join(saving_dir, 'finish'),
            save_word2vec=True,
            save_event_vector=True,
            save_pair_composition=True
        )
