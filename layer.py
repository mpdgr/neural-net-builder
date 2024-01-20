from enum import Enum, auto
from matrix_util import *


class Layer:
    class Location(Enum):
        INPUT = auto()
        HIDDEN = auto()
        OUTPUT = auto()

    weights = None
    prediction = None
    delta = None
    weighted_delta = None
    weighted_delta_alpha = None
    deltas_share = None
    back_deltas = None
    inp = None
    alpha = 0.1

    debug = True

    def __init__(self, node_count, input_count, location, weights="rand", debug=False):
        self.node_count = node_count
        self.input_count = input_count
        # todo: rand weight optional
        self.weights = weights
        self.location = location
        self.debug = debug

    def learning_round(self, inp, target):
        # prediction for each node
        self.comp_prediction(inp)
        # delta for each node
        self.__comp_delta(target)
        # weigh delta to adjust weights
        self.__comp_weight_delta()
        self.__comp_back_deltas()
        self.__adjust_weights()
        # weigh delta to adjust weights
        if self.debug:
            print(f"Back propagate deltas: {self.back_deltas}")
        return self.back_deltas

    # comp prediction basing on input and weights
    # inp size = node_count
    def comp_prediction(self, inp):
        self.inp = inp
        self.prediction = matrix_product(inp, self.weights)
        if self.debug:
            print(f"Prediction: {self.prediction}")
        return self.prediction

    # comp delta = prediction - expected value
    # result size = node_count
    def __comp_delta(self, target):
        self.delta = subtract_vectors(self.prediction, target)
        if self.debug:
            print(f"Delta: {self.delta}")

    # comp weighted delta and apply alpha  = delta * input * alpha
    def __comp_weight_delta(self):
        self.weighted_delta = vector_outer_product(self.inp, self.delta)
        self.weighted_delta_alpha = matrix_scalar_product(self.weighted_delta, self.alpha)
        if self.debug:
            print(f"Weighted delta alpha {self.weighted_delta_alpha}")

    # comp delta shares to back propagate -> delta * weight for each input node connection
    # not required for learning round - logic included in @comp_back_delta_nodes
    def __comp_back_delta_for_weights(self):
        self.deltas_share = vector_matrix_row_wise_product(self.delta, self.weights)
        if self.debug:
            print(f"Delta shares partial: {self.deltas_share}")

    # comp summed deltas to back propagate to each of input nodes
    # -> sum of delta shares for each input nodes
    def __comp_back_deltas(self):
        deltas_share_nodes_matrix = matrix_product(self.weights, transpose_matrix(self.delta))
        self.back_deltas = matrix_to_vector_row_major(deltas_share_nodes_matrix)
        if self.debug:
            print(f"Delta shares for input nodes: {self.back_deltas}")

    # adjust weights - subtract weighted deltas from respective weights
    def __adjust_weights(self):
        if self.debug:
            print(f"Starting weights: {self.weights}")
        self.weights = subtract_matrices(self.weights, self.weighted_delta_alpha)
        if self.debug:
            print(f"Weights after correction weights: {self.weights}")
