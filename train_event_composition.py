from argument_composition import ArgumentCompositionModel
from event_composition import EventCompositionModel, EventCompositionTrainer
import argparse
import os
from utils import PairTuningCorpusIterator, get_console_logger

parser = argparse.ArgumentParser()

parser.add_argument('indexed_corpus',
                    help='Path to the indexed corpus')
parser.add_argument('output_path',
                    help='Path to saving the trained model')
parser.add_argument('--arg_comp_model_path',
                    help='Path to a trained argument composition model that '
                         'gives us our basic event representations that '
                         'we will learn a similarity function of')
parser.add_argument('--event_comp_model_path',
                    help='Path to a partially trained event composition model '
                         'that we will continue training on')
parser.add_argument('--layer-sizes', default='100',
                    help='Comma-separated list of layer sizes (default: 100, '
                         'single layer)')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='Number of examples to include in a minibatch ('
                         'default: 1000)')
parser.add_argument('--tuning-lr', type=float, default=0.025,
                    help='SGD learning rate to use for fine-tuning (default: '
                         '0.025)')
parser.add_argument('--tuning-min-lr', type=float, default=0.0001,
                    help='Learning rate drops off as we go through the '
                         'dataset. Do not let it go lower than this value '
                         '(default: 0.0001)')
parser.add_argument('--tuning-regularization', type=float, default=0.01,
                    help='L2 regularization coefficient for fine-tuning stage '
                         '(default: 0.01)')
parser.add_argument('--tuning-iterations', type=int, default=3,
                    help='Number of iterations to fine tune for (default: 3)')
parser.add_argument('--update-argument-composition', action='store_true',
                    help='Allow fine tuning to adjust the weights that define '
                         'the composition of predicates and arguments into '
                         'an event representation')
parser.add_argument('--update-input-vectors', action='store_true',
                    help='Allow fine tuning to adjust the original word2vec '
                         'vectors, rather than simply learning a composition '
                         'function')
parser.add_argument('--update-empty-vectors', action='store_true',
                    help='Vectors for empty arg slots are initialized to 0. '
                         'Allow these to be learned during training '
                         '(by default, left as 0)')
parser.add_argument('--event-tuning-iterations', type=int, default=0,
                    help='After tuning the event composition function, '
                         'perform some further iterations tuning '
                         'the whole network, including event representations. '
                         'By default, this is not done  at all')

opts = parser.parse_args()

model_name = 'event-comp'

log = get_console_logger('Pair Tuning')

output_dir = os.path.join(opts.output_path, model_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log.info('Started pair tuning')

if opts.arg_comp_model_path:
    log.info('Loading argument composition model from {}'.format(
        opts.arg_comp_model_path))
    arg_comp_model = \
        ArgumentCompositionModel.load_from_directory(opts.arg_comp_model_path)
    # Split up the layer size specification
    layer_sizes = [int(size) for size in opts.layer_sizes.split(',')]
    # Create and initialize the network(s)
    log.info('Initializing network with layer sizes [{}|{}|1]->{}->1'.format(
        arg_comp_model.vector_size, arg_comp_model.vector_size,
        '->'.join(str(s) for s in layer_sizes)))
    model = EventCompositionModel(arg_comp_model, layer_sizes=layer_sizes)
elif opts.event_comp_model_path:
    log.info('Loading partially trained event composition model from {}'.format(
        opts.event_comp_model_path))
    model = EventCompositionModel.load_from_directory(
        opts.event_comp_model_path)
else:
    raise RuntimeError(
        'Must provide either arg_comp_model_path or event_comp_model_path')

model_saving_dir = os.path.join(output_dir, 'init')
log.info('Saving model to {}'.format(model_saving_dir))
model.save_to_directory(model_saving_dir, save_word2vec=False)

# raise error if indexed_corpus doesn't exist
if not os.path.isdir(opts.indexed_corpus):
    log.error('Cannot find indexed corpus at {}'.format(opts.indexed_corpus))
    exit(-1)

corpus_it = PairTuningCorpusIterator(
    opts.indexed_corpus, batch_size=opts.batch_size)
log.info('Found {} lines in the corpus'.format(len(corpus_it)))


def _iteration_callback(iter_num, save_word2vec):
    model_iter_saving_dir = os.path.join(output_dir, 'iter-{}'.format(iter_num))
    log.info('Saving between-iteration model to {}'.format(
        model_iter_saving_dir))
    model.save_to_directory(model_iter_saving_dir, save_word2vec=save_word2vec)


# Start the training algorithm
log.info(
    "Fine tuning with l2 reg={}, lr={}, {}-instance minibatches, "
    "{} its updating composition{}".format(
        opts.tuning_regularization,
        opts.tuning_lr,
        opts.batch_size,
        opts.tuning_iterations,
        "" if not opts.event_tuning_iterations
        else (", %d its updating full network" % opts.event_tuning_iterations)
    )
)

trainer = EventCompositionTrainer(
    model,
    learning_rate=opts.tuning_lr,
    min_learning_rate=opts.tuning_min_lr,
    regularization=opts.tuning_regularization,
    update_argument_composition=opts.update_argument_composition,
    update_input_vectors=opts.update_input_vectors,
    update_empty_vectors=opts.update_empty_vectors
)

# Start the training process
trainer.train(
    corpus_it,
    iterations=opts.tuning_iterations,
    iteration_callback=_iteration_callback,
    log=log
)

model_saving_dir = os.path.join(output_dir, 'finish')
log.info('Saving model to {}'.format(model_saving_dir))
model.save_to_directory(model_saving_dir, save_word2vec=False)

if opts.event_tuning_iterations > 0:
    output_dir = os.path.join(output_dir, 'full-training')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.info(
        "Performing %d iterations updating the full network, including event "
        "representations" %
        opts.event_tuning_iterations)

    model_saving_dir = os.path.join(output_dir, 'init')
    log.info('Saving model to {}'.format(model_saving_dir))
    model.save_to_directory(model_saving_dir, save_word2vec=False)

    full_trainer = EventCompositionTrainer(
        model,
        # Stick at the lowest learning rate
        learning_rate=opts.tuning_min_lr,
        min_learning_rate=opts.tuning_min_lr,
        regularization=opts.tuning_regularization,
        update_argument_composition=True,
        update_input_vectors=True,
        update_empty_vectors=opts.update_empty_vectors
    )

    # start the full training process
    full_trainer.train(
        corpus_it,
        iterations=opts.event_tuning_iterations,
        log=log,
        iteration_callback=_iteration_callback
    )

    model_saving_dir = os.path.join(output_dir, 'finish')
    log.info('Saving model to {}'.format(model_saving_dir))
    model.save_to_directory(model_saving_dir, save_word2vec=False)
