import random
from activation import *
from dropout import *
from util.matrix_util import *
import logging as log


class Layer:
    class Location(Enum):
        INPUT = auto()
        HIDDEN = auto()
        OUTPUT = auto()

    node_count = None
    input_count = None
    location = None
    weights = None
    __prediction = None
    __raw_prediction = None
    __delta = None
    __weighted_delta = None
    __weighted_delta_alpha = None
    __deltas_share = None
    __back_deltas = None
    __inp = None
    __alpha = None
    activation = None
    dropout_rate = 0

    debug = True

    def __init__(self, node_count, input_count, location, activation=None, weights="rand", alpha=0.1, debug=False):
        self.node_count = node_count
        self.input_count = input_count
        self.location = location
        self.activation = activation
        # todo: rand weight optional
        self.weights = weights
        self.__alpha = alpha
        # self.__debug = debug
        if weights == "const":
            self.init_constant_weights(0.5)
        else:
            self.init_random_weights()

    def forward_pass(self, inp, training=False):
        # for input layer input is forwarded directly eventually after applying dropout
        if self.location == Layer.Location.INPUT and training:
            self.__prediction = apply_dropout(self.dropout_rate, inp)
            log.debug(f"Forwarding prediction: {self.__prediction}")
            return self.__prediction
        elif self.location == Layer.Location.INPUT:
            self.__prediction = inp
            log.debug(f"Forwarding prediction: {self.__prediction}")
            return self.__prediction

        # raw prediction for each node
        raw_prediction = self.__predict(inp)
        if training:
            # print("training")
            raw_prediction = apply_dropout(self.dropout_rate, raw_prediction)
        # todo: clean updating predictions
        self.__raw_prediction = raw_prediction
        self.__prediction = raw_prediction
        # apply activation function
        if self.activation is not None:
            self.__prediction = self.activation(raw_prediction, ActivationType.FUNCTION)
        log.debug(f"Forwarding prediction: {self.__prediction}")
        return self.__prediction

    def backward_pass(self, delta):
        self.__delta = delta
        # apply activation function in backpropagation
        if self.activation is not None:
            # compute derivative for input
            derivative_vector = self.activation(self.__raw_prediction, ActivationType.DERIVATIVE)
            # multiply deltas by derivatives
            adjusted_delta = vector_element_wise_product(delta, derivative_vector)
            self.__delta = adjusted_delta
        # weigh delta to adjust weights
        self.__comp_weight_delta()
        self.__comp_back_deltas()
        self.__adjust_weights()
        log.debug(f"Backpropagating deltas: {self.__back_deltas}")
        return self.__back_deltas

    def local_delta(self):
        return

    # comp prediction basing on input and weights, apply activation function
    # inp size = node_count
    def __predict(self, inp):
        self.__inp = inp
        self.__prediction = matrix_product(inp, self.weights)
        log.debug(f"Prediction: {self.__prediction}")
        return self.__prediction

    # comp weighted delta and apply alpha  = delta * input * alpha
    def __comp_weight_delta(self):
        self.__weighted_delta = vector_outer_product(self.__inp, self.__delta)
        self.__weighted_delta_alpha = matrix_scalar_product(self.__weighted_delta, self.__alpha)
        log.debug(f"Weighted delta alpha {self.__weighted_delta_alpha}")

    # comp summed deltas to back propagate to each of input nodes
    # -> sum of delta shares for each input nodes
    def __comp_back_deltas(self):
        deltas_share_nodes_matrix = matrix_product(self.weights, transpose_matrix(self.__delta))
        self.__back_deltas = matrix_to_vector_row_major(deltas_share_nodes_matrix)
        log.debug(f"Delta shares for input nodes: {self.__back_deltas}")

    # adjust weights - subtract weighted deltas from respective weights
    def __adjust_weights(self):
        log.debug(f"Weights before update: {self.weights}")
        self.weights = subtract_matrices(self.weights, self.__weighted_delta_alpha)
        log.debug(f"Weights after update: {self.weights}")

    # node x input
    def init_constant_weights(self, const):
        weights = []
        for r in range(self.input_count):
            row = []
            for n in range(self.node_count):
                row.append(const)
            weights.append(row)

        self.weights = weights
        log.debug(f"Initialized weights: {self.weights}")

    def init_random_weights(self):
        weights = []
        random.seed(12121212)
        for r in range(self.input_count):
            row = []
            for n in range(self.node_count):
                row.append((random.random() - 0.5) * 0.5)
            weights.append(row)

        self.weights = weights
        log.debug(f"Initialized weights: {self.weights}")
