"""
    node_count = 4
    input_count = 3

    node_values = [1, 2, 3, 4]

    weights = [
        [1, 2, 3, 4]
        [1, 2, 3, 4]
        [1, 2, 3, 4]
    ]

    input = [1, 2, 3]
"""
from matrix_util import *

inp = [1, 2, 3]
expected = [20, 15]

weights = [
    [1, 1.5],
    [2, 2.5],
    [3, 3.5]
]


class Layer:
    weights = None
    prediction = None
    delta = None
    weighted_delta = None
    weighted_delta_alpha = None
    deltas_share = None
    deltas_share_nodes = None
    inp = None
    alpha = 0.1

    debug = True

    def __init__(self, node_count, input_count):
        self.node_count = node_count
        self.input_count = input_count
        # todo: rand weight optional
        self.weights = weights

    def learning_round(self, inp, expected):
        # 1. get input - one node count X input count
        # 2. multiply each input by weight and sum to total prediction
        # 3. push totals out
        self.comp_prediction(inp)
        # 4. for each node get its share of delta from next layer
        # 5. received delta is now expected value
        # 6. subtract pred - expected
        # 7. this is new delta
        self.comp_delta(expected)
        self.comp_weight_delta()
        # 8. multiply new delta by input for each incoming input
        self.comp_back_delta()
        self.comp_back_delta_nodes()
        # 14. adjust weights -> dla każdej wagi, waga = waga - result
        self.adjust_weights()
        # 9. multiply result by alpha
        # self.comp_out_delta_alpha()
        # 10. distribute new delta -> multiply by weight
        # 11. the result will be shared across incoming nodes
        # 12. comp share multiplying result by weight for each node
        # 13. share delta between all incoming nodes from previous layer -> multiply
        # self.distribute_delta()

        return self.weights

    # inp size = node_count
    # 1, 2, 3
    def comp_prediction(self, inp):
        self.inp = inp
        self.prediction = matrix_product(inp, self.weights)
        if self.debug:
            print(f"Prediction: {self.prediction}")

        return self.prediction

    # result size = node_count
    # 4, 5, 6, 7
    def comp_delta(self, result):
        self.delta = subtract_vectors(self.prediction, result)
        if self.debug:
            print(f"Delta: {self.delta}")

    def comp_weight_delta(self):
        self.weighted_delta = vector_outer_product(inp, self.delta)
        self.weighted_delta_alpha = matrix_scalar_product(self.weighted_delta, self.alpha)
        if self.debug:
            print(f"Weighted delta alpha {self.weighted_delta_alpha}")

    # 8
    def comp_back_delta(self):
        self.deltas_share = vector_matrix_row_wise_product(self.delta, self.weights)
        if self.debug:
            print(f"Delta shares: {self.deltas_share}")

    def comp_back_delta_nodes(self):
        deltas_share_nodes_matrix = matrix_product(self.weights, transpose_matrix(self.delta))
        self.deltas_share_nodes = matrix_to_vector_row_major(deltas_share_nodes_matrix)
        if self.debug:
            print(f"Delta shares input nodes: {self.deltas_share_nodes}")

    # 14
    def adjust_weights(self):
        if self.debug:
            print(f"Starting weights: {self.weights}")
        self.weights = subtract_matrices(self.weights, self.weighted_delta_alpha)
        if self.debug:
            print(f"Weights after correction weights: {self.weights}")

    def learning_round(self, inp, expected):
        # 1. get input - one node count X input count
        # 2. multiply each input by weight and sum to total prediction
        # 3. push totals out
        self.comp_prediction(inp)
        # 4. for each node get its share of delta from next layer
        # 5. received delta is now expected value
        # 6. subtract pred - expected
        # 7. this is new delta
        self.comp_delta(expected)
        self.comp_weight_delta()
        # 8. multiply new delta by input for each incoming input
        self.comp_back_delta()
        self.comp_back_delta_nodes()
        # 14. adjust weights -> dla każdej wagi, waga = waga - result
        self.adjust_weights()
        # 9. multiply result by alpha
        # self.comp_out_delta_alpha()
        # 10. distribute new delta -> multiply by weight
        # 11. the result will be shared across incoming nodes
        # 12. comp share multiplying result by weight for each node
        # 13. share delta between all incoming nodes from previous layer -> multiply
        # self.distribute_delta()

        return self.weights


layer = Layer(2, 3)
layer.weights = weights
# print(matrix_product(weights, weights))
print(layer.learning_round(inp, expected))
