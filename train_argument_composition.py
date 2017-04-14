from argument_composition import ArgumentCompositionModel
from argument_composition import EventVectorNetwork
from autoencoder import DenoisingAutoencoderIterableTrainer
from word2vec import Word2VecModel
from utils import PretrainingCorpusIterator, get_console_logger
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("word2vec_vector",
                    help="Path of a trained word2vec vector file")
parser.add_argument("word2vec_vocab",
                    help="Path of a trained word2vec vocabulary file")
parser.add_argument("indexed_corpus",
                    help="Path to the indexed corpus")
parser.add_argument("output",
                    help="Path to saving the trained model")
parser.add_argument("--layer-sizes", default="100",
                    help="Comma-separated list of layer sizes "
                         "(default: 100, single layer)")
parser.add_argument("--corruption", type=float, default=0.2,
                    help="Level of drop-out noise to apply during "
                         "training, 0.0-1.0 (default: 0.2)")
parser.add_argument("--batch-size", type=int, default=1000,
                    help="Number of examples to include in a minibatch "
                         "(default: 1000)")
parser.add_argument("--iterations", type=int, default=10,
                    help="Number of iterations to train each layer for "
                         "(default: 10)")
parser.add_argument("--regularization", type=float, default=0.01,
                    help="L2 regularization coefficient "
                         "(default: 0.01)")
parser.add_argument("--lr", type=float, default=0.1,
                    help="SGD learning rate (default: 0.1)")


opts = parser.parse_args()

model_name = "arg-comp"

log = get_console_logger("Autoencoder Pretraining")

log.info("Started autoencoder pretraining")

# Split up the layer size specification
layer_sizes = [int(size) for size in opts.layer_sizes.split(",")]

# Load the word2vec model that we're using as input
log.info("Loading base word2vec model")
word2vec_model = Word2VecModel.load_model(
    opts.word2vec_vector, opts.word2vec_vocab)

# Create and initialize the network(s)
log.info("Initializing network with layer sizes %d->%s" %
         (4 * word2vec_model.dimension,
          "->".join(str(s) for s in layer_sizes)))

network = EventVectorNetwork(
    word2vec_model.get_vector_matrix(), layer_sizes=layer_sizes)
model = ArgumentCompositionModel(network, word2vec_model.get_vocab())
model.save_to_directory(opts.output)

# raise error if indexed_corpus doesn't exist
if not os.path.isdir(opts.indexed_corpus):
    log.error("Cannot find indexed corpus at {}".format(opts.indexed_corpus))
    exit(-1)

# Normal, autoencoder pre-training
# Start the training algorithm
log.info("Pretraining with l2 reg=%f, lr=%f, corruption=%f, "
         "%d iterations per layer, %d-instance minibatches" %
         (opts.regularization, opts.lr, opts.corruption,
          opts.iterations, opts.batch_size))
for layer in range(len(layer_sizes)):
    log.info("Pretraining layer %d" % layer)
    corpus_it = PretrainingCorpusIterator(opts.indexed_corpus, model, layer,
                                          batch_size=opts.batch_size)
    trainer = DenoisingAutoencoderIterableTrainer(network.layers[layer])
    trainer.train(
        corpus_it,
        iterations=opts.iterations,
        log=log,
        learning_rate=opts.lr,
        regularization=opts.regularization,
        corruption_level=opts.corruption,
        loss="l2",
    )

    log.info("Finished training layer %d" % layer)
    log.info("Saving model: %s" % model_name)
    model.save_to_directory(model_name)

log.info("Finished autoencoder pretraining")
model.save_to_directory(model_name)
