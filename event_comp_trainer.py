import argparse
import os

from event_comp_model import EventCompositionModel, EventCompositionTrainer
from rich_script import PairTuningCorpusIterator
from util import Word2VecModel, consts, get_console_logger

parser = argparse.ArgumentParser()

parser.add_argument('indexed_corpus',
                    help='Path to the indexed corpus for training')
parser.add_argument('output_path',
                    help='Path to saving the trained model')
parser.add_argument('stage', type=int,
                    help='Stage in training EventCompositionModel, '
                         '1 = autoencoder pre-training, '
                         '2 = pair composition fine tuning, '
                         '3 = pair composition full fine tuning '
                         '(updating event vector network)')
parser.add_argument('--word2vec_vector',
                    help='Path of a trained word2vec vector file, '
                         'only used in stage 1')
parser.add_argument('--word2vec_vocab',
                    help='Path of a trained word2vec vocabulary file, '
                         'only used in stage 2')
parser.add_argument('--layer_sizes', default='100',
                    help='Comma-separated list of layer sizes '
                         '(default: 100, single layer), only used in '
                         'stage 1 (initializing event vector network)'
                         'or stage 2 (initializing pair composition network')
parser.add_argument('--corruption', type=float, default=0.2,
                    help='Level of drop-out noise to apply during '
                         'autoencoder pre-training, 0.0-1.0 (default: 0.2), '
                         'only used in stage 1')
parser.add_argument('--input_path',
                    help='Path to load a partially trained model, '
                         'only used in stage 2/3')
parser.add_argument('--val_indexed_corpus',
                    help='Path to the indexed corpus for validation, '
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

if opts.stage == 1:
    log.info('Loading pre-trained word2vec model from {} and {}'.format(
        opts.word2vec_vector, opts.word2vec_vocab))
    word2vec = Word2VecModel.load_model(opts.word2vec_vector,
                                        fvocab=opts.word2vec_vocab)

    layer_sizes = [int(size) for size in opts.layer_sizes.split(',')]

    log.info('Initializing event vector network with layer sizes {}->{}'.format(
        4 * word2vec.vector_size, '->'.join(str(s) for s in layer_sizes)))

    event_composition_model = EventCompositionModel(
        word2vec, event_vector_layer_sizes=layer_sizes)

    event_composition_trainer = EventCompositionTrainer(
        event_composition_model, saving_path=opts.output_path, log=log)

    event_composition_trainer.autoencoder_pretraining(
        indexed_corpus=opts.indexed_corpus,
        batch_size=opts.batch_size,
        iterations=opts.iterations,
        learning_rate=opts.lr,
        regularization=opts.regularization,
        corruption_level=opts.corruption
    )

elif opts.stage == 2 or opts.stage == 3:
    log.info('Loading partially trained model from {}'.format(opts.input_path))
    event_composition_model = EventCompositionModel.load_model(opts.input_path)

    assert event_composition_model.event_vector_network, \
        'event_vector_network in the model cannot be None'

    if opts.stage == 2:
        if event_composition_model.pair_composition_network is None:
            layer_sizes = [int(size) for size in opts.layer_sizes.split(',')]
            log.info(
                'Initializing pair composition network with layer sizes '
                '[{0}|{0}|1(arg_idx)|{1}(salience)]->{2}->1'.format(
                    event_composition_model.event_vector_network.vector_size,
                    consts.NUM_SALIENCE_FEATURES,
                    '->'.join(str(s) for s in layer_sizes)))
            event_composition_model.add_pair_projection_network(layer_sizes)
    else:
        assert event_composition_model.pair_composition_network, \
            'pair_composition_network in the model cannot be None'

    event_composition_trainer = EventCompositionTrainer(
        event_composition_model, saving_path=opts.output_path, log=log)

    if not os.path.isdir(opts.indexed_corpus):
        log.error(
            'Cannot find indexed corpus at {}'.format(opts.indexed_corpus))
        exit(-1)

    log.info(
        'Loading indexed corpus from: {}, with batch_size={}'.format(
            opts.indexed_corpus, opts.batch_size))
    corpus_it = PairTuningCorpusIterator(
        opts.indexed_corpus, batch_size=opts.batch_size)
    log.info('Found {} lines in the corpus'.format(len(corpus_it)))

    val_corpus_it = None
    if opts.val_indexed_corpus and os.path.isdir(opts.val_indexed_corpus):
        log.info(
            'Loading validation indexed corpus from: {}, '
            'with batch_size={}'.format(
                opts.val_indexed_corpus, opts.batch_size))
        val_corpus_it = PairTuningCorpusIterator(
            opts.val_indexed_corpus, batch_size=opts.batch_size)
        log.info('Found {} lines in the corpus'.format(len(corpus_it)))

    if opts.stage == 2:
        event_composition_trainer.fine_tuning(
            batch_iterator=corpus_it,
            iterations=opts.iterations,
            learning_rate=opts.lr,
            min_learning_rate=opts.min_lr,
            regularization=opts.regularization,
            update_event_vectors=False,
            update_input_vectors=False,
            update_empty_vectors=False,
            val_batch_iterator=val_corpus_it
        )
    else:
        event_composition_trainer.fine_tuning(
            batch_iterator=corpus_it,
            iterations=opts.iterations,
            learning_rate=opts.lr,
            min_learning_rate=opts.lr,  # set min_lr = lr in full fine tuning
            regularization=opts.regularization,
            update_event_vectors=True,
            update_input_vectors=True,
            update_empty_vectors=opts.update_empty_vectors,
            val_batch_iterator=val_corpus_it
        )

else:
    raise ValueError(
        'Unrecognized stage parameter: {}, expecting 1, 2 or 3'.format(
            opts.stage))
