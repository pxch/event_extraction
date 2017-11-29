import argparse
from os import makedirs
from os.path import exists, isdir, join

from event_comp_model import EventCompositionModel, EventCompositionTrainer
from rich_script import PairTuningCorpusIterator
from util import get_console_logger

parser = argparse.ArgumentParser()

parser.add_argument('indexed_corpus',
                    help='Path to the indexed corpus for training')
parser.add_argument('output_path',
                    help='Path to saving the trained model')
parser.add_argument('--num_splits', type=int, default=10,
                    help='Number of splits in cross validation')
parser.add_argument('--input_path',
                    help='Path to load a partially trained model, '
                         'only used in stage 2/3')
parser.add_argument('--iterations', type=int, default=10,
                    help='Number of training iterations (default: 10)')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of examples to include in a minibatch '
                         '(default: 100)')
parser.add_argument('--regularization', type=float, default=0.01,
                    help='L2 regularization coefficient (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='SGD learning rate (default: 0.1)')
parser.add_argument('--min_lr', type=float, default=0.01,
                    help='Minimum SGD learning rate to drop off '
                         '(default: 0.01), only used in stage 2')
parser.add_argument('--update_empty_vectors', action='store_true',
                    help='Vectors for empty arg slots are initialized to 0. '
                         'Allow these to be learned during full fine tuning. '
                         'only used in stage 3')

opts = parser.parse_args()

log = get_console_logger('event_comp_trainer')

for i in range(opts.num_splits):
    print 'Training fold #{}'.format(i)
    print 'Loading partially trained model from {}'.format(opts.input_path)

    event_composition_model = EventCompositionModel.load_model(opts.input_path)

    assert event_composition_model.event_vector_network, \
        'event_vector_network in the model cannot be None'
    assert event_composition_model.pair_composition_network, \
        'pair_composition_network in the model cannot be None'

    use_salience = \
        event_composition_model.pair_composition_network.use_salience
    salience_features = \
        event_composition_model.pair_composition_network.salience_features

    output_path = join(opts.output_path, str(i))
    if not exists(output_path):
        makedirs(output_path)

    event_composition_trainer = EventCompositionTrainer(
        event_composition_model, saving_path=output_path, log=log)

    training_path = join(opts.indexed_corpus, str(i), 'training')
    if not isdir(training_path):
        log.error(
            'Cannot find indexed corpus at {}'.format(training_path))
        exit(-1)

    log.info(
        'Loading indexed corpus from: {}, with batch_size={}, '
        'use_salience={}, salience_features={}'.format(
            training_path, opts.batch_size, use_salience, salience_features))
    corpus_it = PairTuningCorpusIterator(
        training_path, batch_size=opts.batch_size,
        use_salience=use_salience, salience_features=salience_features)
    log.info('Found {} lines in the corpus'.format(len(corpus_it)))

    validation_path = join(opts.indexed_corpus, str(i), 'validation')
    if not isdir(validation_path):
        log.error(
            'Cannot find indexed corpus at {}'.format(validation_path))
        exit(-1)

    log.info(
        'Loading validation indexed corpus from: {}, with batch_size={}, '
        'use_salience={}, salience_features={}'.format(
            validation_path, opts.batch_size, use_salience, salience_features))
    val_corpus_it = PairTuningCorpusIterator(
        validation_path, batch_size=opts.batch_size,
        use_salience=use_salience, salience_features=salience_features)
    log.info('Found {} lines in the corpus'.format(len(val_corpus_it)))

    event_composition_trainer.fine_tuning(
        batch_iterator=corpus_it,
        iterations=opts.iterations,
        learning_rate=opts.lr,
        min_learning_rate=opts.min_lr,
        regularization=opts.regularization,
        update_event_vectors=True,
        update_input_vectors=True,
        update_empty_vectors=opts.update_empty_vectors,
        val_batch_iterator=val_corpus_it
    )