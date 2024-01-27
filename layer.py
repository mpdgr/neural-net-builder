from enum import Enum, auto
from util.matrix_util import *


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
    __delta = None
    __weighted_delta = None
    __weighted_delta_alpha = None
    __deltas_share = None
    __back_deltas = None
    __inp = None
    __alpha = None

    debug = True

    def __init__(self, node_count, input_count, location, weights="rand", alpha=0.1, debug=False):
        self.node_count = node_count
        self.input_count = input_count
        self.location = location
        # todo: rand weight optional
        self.weights = weights
        self.__alpha = alpha
        # self.__debug = debug
        if weights == "rand":
            self.init_constant_weights(0.5)

    def forward_pass(self, inp):
        # prediction for each node
        prediction = self.__predict(inp)
        if self.debug:
            print(f"Forwarding prediction: {prediction}")
        return self.__prediction

    # used for middle layers
    def backward_pass(self, delta):
        self.__delta = delta
        # weigh delta to adjust weights
        self.__comp_weight_delta()
        self.__comp_back_deltas()
        self.__adjust_weights()
        if self.debug:
            print(f"Backpropagating deltas: {self.__back_deltas}")
        return self.__back_deltas

    # used for input layer
    def _consume_delta(self, delta):
        if self.debug:
            print(f"Input layer received deltas: {self.__back_deltas}")

    # used for output layer
    def output_round(self, inp, target):
        # prediction for each node
        self.__predict(inp)
        # delta for each node
        self.__comp_delta(target)
        # weigh delta to adjust weights
        self.__comp_weight_delta()
        self.__comp_back_deltas()
        self.__adjust_weights()
        if self.debug:
            print(f"Back propagate deltas: {self.__back_deltas}")
        return self.__back_deltas

    # used for input layer
    def _input_pass(self, inp):
        self.__inp = inp
        return inp

    # comp prediction basing on input and weights
    # inp size = node_count
    def __predict(self, inp):
        self.__inp = inp
        self.__prediction = matrix_product(inp, self.weights)
        if self.debug:
            print(f"Prediction: {self.__prediction}")
        return self.__prediction

    # comp delta = prediction - expected value
    # result size = node_count
    def __comp_delta(self, target):
        self.__delta = subtract_vectors(self.__prediction, target)
        if self.debug:
            print(f"Delta: {self.__delta}")

    # comp weighted delta and apply alpha  = delta * input * alpha
    def __comp_weight_delta(self):
        self.__weighted_delta = vector_outer_product(self.__inp, self.__delta)
        self.__weighted_delta_alpha = matrix_scalar_product(self.__weighted_delta, self.__alpha)
        if self.debug:
            print(f"Weighted delta alpha {self.__weighted_delta_alpha}")

    # comp delta shares to back propagate -> delta * weight for each input node connection
    # not required for learning round - logic included in @comp_back_delta_nodes
    def __comp_back_delta_for_weights(self):
        self.__deltas_share = vector_matrix_row_wise_product(self.__delta, self.weights)
        if self.debug:
            print(f"Delta shares partial: {self.__deltas_share}")

    # comp summed deltas to back propagate to each of input nodes
    # -> sum of delta shares for each input nodes
    def __comp_back_deltas(self):
        deltas_share_nodes_matrix = matrix_product(self.weights, transpose_matrix(self.__delta))
        self.__back_deltas = matrix_to_vector_row_major(deltas_share_nodes_matrix)
        if self.debug:
            print(f"Delta shares for input nodes: {self.__back_deltas}")

    # adjust weights - subtract weighted deltas from respective weights
    def __adjust_weights(self):
        if self.debug:
            print(f"Starting weights: {self.weights}")
        self.weights = subtract_matrices(self.weights, self.__weighted_delta_alpha)
        if self.debug:
            print(f"Weights after correction weights: {self.weights}")

    # node x input
    def init_constant_weights(self, const):
        weights = []
        for r in range(self.input_count):
            row = []
            for n in range(self.node_count):
                row.append(const)
            weights.append(row)

        self.weights = weights
        if self.debug:
            print(f"Initialized weights: {self.weights}")
