import numpy
import theano
import theano.tensor as T

from pair_composition_network import PairCompositionNetwork
from util import get_class_name, get_console_logger


class PairCompositionTrainer(object):
    def __init__(self, model, learning_rate=0.025, min_learning_rate=0.0001,
                 regularization=0.01, update_event_vectors=False,
                 update_input_vectors=False, update_empty_vectors=False):
        assert isinstance(model, PairCompositionNetwork), \
            'model must be a {} instance'.format(
                get_class_name(PairCompositionNetwork))
        self.model = model

        self.min_learning_rate = min_learning_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
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

        if update_event_vectors:
            # Add the event-internal composition weights
            self.params.extend(
                sum([[layer.W, layer.b] for layer
                     in self.model.event_vector_network.layers], []))
            self.regularized_params.extend(
                [layer.W for layer in self.model.event_vector_network.layers])

        if update_input_vectors:
            self.params.append(self.model.event_vector_network.vectors)

        if update_empty_vectors:
            self.params.extend([
                self.model.event_vector_network.empty_subj_vector,
                self.model.event_vector_network.empty_obj_vector,
                self.model.event_vector_network.empty_pobj_vector
            ])

        self.update_event_vectors = update_event_vectors
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
              log_every_batch=1000):
        # TODO: add logic for validation set and stopping_iterations parameter
        if log is None:
            log = get_console_logger("Pair tuning")

        log.info("Tuning params: learning rate=%s (->%s), regularization=%s" %
                 (self.learning_rate, self.min_learning_rate,
                  self.regularization))
        if self.update_event_vectors:
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
                        'Iteration {}: Processed {:>8d}/{:>8d} batches, '
                        'learning rate = {:g}'.format(
                            i, batch_num + 1, batch_iterator.num_batch,
                            learning_rate))

            log.info(
                'Iteration {}: Processed {:>8d}/{:>8d} batches'.format(
                    i, batch_iterator.num_batch, batch_iterator.num_batch))

            training_costs.append(err / (batch_num+1))

            log.info("COMPLETED ITERATION %d: training cost=%.5g" %
                     (i, training_costs[-1]))

            if iteration_callback is not None:
                # Not computing training error at the moment
                iteration_callback(i)

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
