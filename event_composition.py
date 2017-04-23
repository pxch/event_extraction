from autoencoder import DenoisingAutoencoder
from argument_composition import ArgumentCompositionModel
import numpy
import theano
import theano.tensor as T
import utils
import os
import pickle


class EventCompositionTrainer(object):
    def __init__(self, model, learning_rate=0.025, min_learning_rate=0.0001,
                 regularization=0.01, update_argument_composition=False,
                 update_input_vectors=False, update_empty_vectors=False):
        self.min_learning_rate = min_learning_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.model = model
        self.learning_rate_var = T.scalar(
            "learning_rate", dtype=theano.config.floatX)

        # Collect parameters to be tuned from all layers
        self.params = []
        self.regularized_params = []
        # Always add our own between-event composition function weights
        self.params.extend(
            sum([[layer.W, layer.b] for layer in model.layers], [])
        )
        self.regularized_params.extend([layer.W for layer in model.layers])

        self.params.append(self.model.prediction_weights)
        self.params.append(self.model.prediction_bias)
        self.regularized_params.append(self.model.prediction_weights)

        if update_argument_composition:
            # Add the event-internal composition weights
            self.params.extend(
                sum([[layer.W, layer.b] for layer
                     in self.model.arg_comp_model.layers], []))
            self.regularized_params.extend(
                [layer.W for layer in self.model.arg_comp_model.layers])

        if update_input_vectors:
            self.params.append(self.model.arg_comp_model.vectors)

        if update_empty_vectors:
            self.params.extend([
                self.model.arg_comp_model.empty_subj_vector,
                self.model.arg_comp_model.empty_obj_vector,
                self.model.arg_comp_model.empty_pobj_vector
            ])

        self.update_argument_composition = update_argument_composition
        self.update_input_vectors = update_input_vectors
        self.update_empty_vectors = update_empty_vectors

        self.positive = T.vector("positive", dtype="int8")
        self.positive_col = self.positive.dimshuffle(0, 'x')

    def get_triple_cost_updates(self, regularization=None):
        if regularization is None:
            regularization = self.regularization

        # Compute the two similarities predicted by our current composition
        pos_coherences, neg_coherences = self.model.get_coherence_pair()
        # We want pos coherences to be higher than neg
        # Try to make pos - neg as high as possible (it ranges between -1 and 1)
        cost_per_input = -T.log(pos_coherences) - T.log(1. - neg_coherences)
        cost = T.mean(cost_per_input)

        if regularization > 0.:
            # Collect weights to be regularized from all layers
            reg_term = regularization * T.sum(
                [T.sum(w ** 2) for w in self.regularized_params])
            # Turn this into a mean by counting up the weight params
            reg_term /= T.cast(
                T.sum([T.prod(T.shape(w)) for w in self.regularized_params]),
                theano.config.floatX)
            cost += reg_term

        # Now differentiate to get the updates
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - self.learning_rate_var * gparam)
                   for param, gparam in zip(self.params, gparams)]

        return cost, updates

    def train(self, batch_iterator, iterations=10000, iteration_callback=None,
              log=None, training_cost_prop_change_threshold=0.0005,
              log_every_batch=100):
        if log is None:
            log = utils.get_console_logger("Pair tuning")

        log.info("Tuning params: learning rate=%s (->%s), regularization=%s" %
                 (self.learning_rate, self.min_learning_rate,
                  self.regularization))
        if self.update_argument_composition:
            log.info("Updating argument composition model")
        if self.update_input_vectors:
            log.info("Updating basic word representations")
        if self.update_empty_vectors:
            log.info("Training empty vectors")

        # Compile functions
        # Prepare cost/update functions for training
        cost, updates = self.get_triple_cost_updates()
        # Prepare training functions
        train_fn = theano.function(
            inputs=self.model.triple_inputs + [
                # Allow the learning rate to be set per update
                theano.In(self.learning_rate_var, value=self.learning_rate)
            ],
            outputs=cost,
            updates=updates,
        )

        # Keep a record of costs, so we can plot them
        training_costs = []

        below_threshold_its = 0

        for i in range(iterations):
            err = 0.0
            batch_num = 0
            if i == 0:
                learning_rate = self.learning_rate
            else:
                learning_rate = self.min_learning_rate

            for batch_num, batch_inputs in enumerate(batch_iterator):
                # Shuffle the training data between iterations, as one should
                # with SGD
                # Just shuffle within batches
                shuffle = numpy.random.permutation(batch_inputs[0].shape[0])
                for batch_data in batch_inputs:
                    batch_data[:] = batch_data[shuffle]

                # Update the model with this batch's data
                err += train_fn(*batch_inputs, learning_rate=learning_rate)

                # Update the learning rate, so it falls away as we go through
                # Do this only on the first iteration
                # After that, LR should just stay at the min
                if i == 0:
                    learning_rate = max(
                        self.min_learning_rate,
                        self.learning_rate * (1. - float(batch_num + 1) /
                                              batch_iterator.num_batch))

                if (batch_num + 1) % log_every_batch == 0:
                    log.info(
                        'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                            i, batch_num + 1, batch_iterator.num_batch))
                    if i == 0:
                        log.info("Learning rate updated to %g" % learning_rate)

            log.info(
                'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                    i, batch_iterator.num_batch, batch_iterator.num_batch))

            training_costs.append(err / (batch_num+1))

            log.info("COMPLETED ITERATION %d: training cost=%.5g" %
                     (i, training_costs[-1]))

            if iteration_callback is not None:
                # Not computing training error at the moment
                iteration_callback(i, self.update_input_vectors)

            # Check the proportional change between this iteration's training
            # cost and the last
            if len(training_costs) > 2:
                training_cost_prop_change = abs(
                    (training_costs[-2] - training_costs[-1]) /
                    training_costs[-2])
                if training_cost_prop_change < \
                        training_cost_prop_change_threshold:
                    # Very small change in training cost - maybe we've converged
                    below_threshold_its += 1
                    if below_threshold_its >= 5:
                        # We've had enough iterations with very small changes:
                        # we've converged
                        log.info(
                            "Proportional change in training cost (%g) below "
                            "%g for five successive iterations: converged" %
                            (training_cost_prop_change,
                             training_cost_prop_change_threshold))
                        break
                    else:
                        log.info(
                            "Proportional change in training cost (%g) below "
                            "%g for %d successive iterations: "
                            "waiting until it's been low for five iterations" %
                            (training_cost_prop_change,
                             training_cost_prop_change_threshold,
                             below_threshold_its))
                else:
                    # Reset the below threshold counter
                    below_threshold_its = 0


class EventCompositionModel(object):
    def __init__(self, arg_comp_model, layer_sizes):
        self.arg_comp_model = arg_comp_model
        self.input_a, self.input_b = self.arg_comp_model.get_projection_pair()
        self.input_arg_type = T.vector("arg_type", dtype="int32")
        self.input_vector = T.concatenate(
            (self.input_a, self.input_b,
             self.input_arg_type.dimshuffle(0, 'x')), axis=1)
        self.layer_sizes = layer_sizes

        # Initialize each layer as an autoencoder,
        # allowing us to initialize it by pretraining
        self.layers = []
        self.layer_outputs = []
        input_size = self.arg_comp_model.layer_sizes[-1] * 2 + 1
        layer_input = self.input_vector
        for layer_size in layer_sizes:
            self.layers.append(
                DenoisingAutoencoder(
                    input=layer_input,
                    n_visible=input_size,
                    n_hidden=layer_size,
                    non_linearity="tanh"
                )
            )
            input_size = layer_size
            layer_input = self.layers[-1].hidden_layer
            self.layer_outputs.append(layer_input)
        self.final_projection = layer_input

        # Add a final layer, which will only ever be trained with
        # a supervised objective
        # This is simply a logistic regression layer to predict
        # a coherence score for the input pair
        self.prediction_weights = theano.shared(
            # Just initialize to zeros, so we start off predicting 0.5
            # for every input
            numpy.asarray(
                numpy.random.uniform(
                    low=2. * -numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    high=2. * numpy.sqrt(6. / (layer_sizes[-1] + 1)),
                    size=(layer_sizes[-1], 1),
                ),
                dtype=theano.config.floatX
            ),
            name="prediction_w",
            borrow=True
        )
        self.prediction_bias = theano.shared(
            value=numpy.zeros(1, dtype=theano.config.floatX),
            name="prediction_b",
            borrow=True
        )
        self.prediction = T.nnet.sigmoid(
            T.dot(self.final_projection, self.prediction_weights) +
            self.prediction_bias
        )

        self.pair_inputs = [
            self.arg_comp_model.pred_input_a, self.arg_comp_model.subj_input_a,
            self.arg_comp_model.obj_input_a, self.arg_comp_model.pobj_input_a,
            self.arg_comp_model.pred_input_b, self.arg_comp_model.subj_input_b,
            self.arg_comp_model.obj_input_b, self.arg_comp_model.pobj_input_b,
            self.input_arg_type
        ]
        self.triple_inputs = [
            self.arg_comp_model.pred_input_a, self.arg_comp_model.subj_input_a,
            self.arg_comp_model.obj_input_a, self.arg_comp_model.pobj_input_a,
            self.arg_comp_model.pred_input_b, self.arg_comp_model.subj_input_b,
            self.arg_comp_model.obj_input_b, self.arg_comp_model.pobj_input_b,
            self.arg_comp_model.pred_input_c, self.arg_comp_model.subj_input_c,
            self.arg_comp_model.obj_input_c, self.arg_comp_model.pobj_input_c,
            self.input_arg_type
        ]

        self._coherence_fn = None

    @property
    def coherence_fn(self):
        if self._coherence_fn is None:
            self._coherence_fn = theano.function(
                inputs=self.pair_inputs,
                outputs=self.prediction,
                name="pair_coherence",
            )
        return self._coherence_fn

    def get_coherence_pair(self):
        # Clone prediction function so we can perform two predictions
        # in the same step
        coherence_a = self.prediction

        # Replace b inputs with c inputs
        input_replacements = dict(zip(self.triple_inputs[4:8],
                                      self.triple_inputs[8:12]))
        coherence_b = theano.clone(self.prediction, replace=input_replacements)

        return coherence_a, coherence_b

    def get_layer_input_function(self, layer_num):
        if layer_num >= len(self.layers):
            raise ValueError(
                "cannot get input function for layer %d in a %d-layer network" %
                (layer_num, len(self.layers)))
        elif layer_num == 0:
            # The input to the first layer is just the concatenated input vector
            output_eq = self.input_vector
        else:
            # Otherwise it's the output from the previous layer
            output_eq = self.layer_outputs[layer_num-1]

        return theano.function(
            inputs=self.pair_inputs,
            outputs=output_eq,
            name="layer-%d-input" % layer_num,
        )

    def copy_coherence_function(self, input_a=None, input_b=None,
                                input_arg_type=None):
        """
        Build a new coherence function, copying all weights and such from
        this network, replacing components given as kwargs. Note that this
        uses the same shared variables and any other non-replaced components
        as the network's original expression graph: bear in mind if you use
        it to update weights or combine with other graphs.

        """
        input_a = input_a or self.input_a
        input_b = input_b or self.input_b
        input_arg_type = input_arg_type or self.input_arg_type

        # Build a new coherence function, combining these two projections
        input_vector = T.concatenate(
            [input_a, input_b, input_arg_type], axis=input_a.ndim-1)

        # Initialize each layer as an autoencoder.
        # We'll then set its weights and never use it as an autoencoder
        layers = []
        layer_outputs = []
        input_size = self.arg_comp_model.layer_sizes[-1] * 2
        layer_input = input_vector
        for layer_size in self.layer_sizes:
            layers.append(
                DenoisingAutoencoder(
                    input=layer_input,
                    n_visible=input_size,
                    n_hidden=layer_size,
                    non_linearity="tanh"
                )
            )
            input_size = layer_size
            layer_input = layers[-1].hidden_layer
            layer_outputs.append(layer_input)
        final_projection = layer_input

        # Set the weights of all layers to the ones trained in the base network
        for layer, layer_weights in zip(layers, self.get_weights()):
            layer.set_weights(layer_weights)

        # Add a final layer
        # This is simply a logistic regression layer to predict
        # a coherence score for the input pair
        activation = T.dot(final_projection, self.prediction_weights) + \
                     self.prediction_bias
        # Remove the last dimension, which should now just be of size 1
        activation = activation.reshape(activation.shape[:-1],
                                        ndim=activation.ndim-1)
        prediction = T.nnet.sigmoid(activation)

        return prediction, input_vector, layers, layer_outputs, activation

    def get_weights(self):
        return [ae.get_weights() for ae in self.layers] + \
               [self.prediction_weights.get_value(),
                self.prediction_bias.get_value()]

    def set_weights(self, weights):
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)
        self.prediction_weights.set_value(weights[len(self.layers)])
        self.prediction_bias.set_value(weights[len(self.layers) + 1])

    @classmethod
    def load_from_directory(cls, directory):
        if not os.path.exists(directory):
            raise RuntimeError("{} doesn't exist, abort".format(directory))

        arg_comp_model = ArgumentCompositionModel.load_from_directory(
            os.path.join(directory, 'arg-comp'))

        with open(os.path.join(directory, "layer_sizes"), "r") as f:
            layer_sizes = pickle.load(f)
        with open(os.path.join(directory, "weights"), "r") as f:
            weights = pickle.load(f)
        model = cls(arg_comp_model, layer_sizes=layer_sizes)
        model.set_weights(weights)
        return model

    def save_to_directory(self, directory, save_word2vec=False):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.arg_comp_model.save_to_directory(
            os.path.join(directory, 'arg-comp'), save_word2vec=save_word2vec)

        with open(os.path.join(directory, "weights"), "w") as f:
            pickle.dump(self.get_weights(), f)
        with open(os.path.join(directory, "layer_sizes"), "w") as f:
            pickle.dump(self.layer_sizes, f)
