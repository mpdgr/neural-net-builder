import dropout
import logging as log

from layer import Layer
from util.matrix_util import subtract_vectors
from activation import *


class Network:
    # layers array
    layers = []

    # dropout array -> dropout rate at each node
    # dropout array length must equal layers array - 1 (no dropout on output layer)
    dropout = []

    # activation array
    # activation array length must equal layers array - 1 (no activation on input layer)
    activation = []

    layers_count = None
    debug = False

    def __init__(self, nodes, dropout=None, activation=None, debug=False):
        layers_count = len(nodes)
        if layers_count < 2:
            print("Network must include at lease two layers")
            return

        log.info(f"Initializing network - layers count: {layers_count}")

        # init layers
        layers = []
        for i in range(layers_count):
            if i == 0:
                layers.append(Layer(nodes[0], 0, Layer.Location.INPUT))
            elif i < layers_count - 1:
                layers.append(Layer(nodes[i], nodes[i - 1], Layer.Location.HIDDEN))
            else:
                layers.append(Layer(nodes[i], nodes[i - 1], Layer.Location.OUTPUT))

        # set layers dropouts
        if dropout is not None:
            if len(dropout) != len(layers) - 1:
                print("Dropout array length must equal layers array - 1")
                return
            else:
                for i in range(layers_count - 1):
                    layers[i].dropout_rate = dropout[i]

        # set activation functions
        if activation is not None:
            if len(activation) != len(layers) - 1:
                print("Activation array length must equal layers array - 1")
                return
            else:
                layers[0].activation = none
                for i in range(layers_count - 1):
                    layers[i + 1].activation = activation[i]
        else:
            for i in range(layers_count):
                layers[i].activation = none

        self.layers = layers
        self.layers_count = layers_count
        self.dropout = dropout
        self.activation = activation
        self.debug = debug

        layers_info = [f"[{layer.input_count}, {layer.node_count}]" for layer in self.layers]
        log.info(f"Initialized network with layers: {', '.join(layers_info)}")

    # for inference phase set training to false
    def predict(self, inp, training=False):
        prediction = inp
        for i in range(0, self.layers_count):
            prediction = self.layers[i].forward_pass(prediction, training)
        return prediction

    def back_propagate(self, deltas):
        next_deltas = deltas
        for i in range(0, self.layers_count - 1):
            next_deltas = self.layers[self.layers_count - i - 1].backward_pass(next_deltas)
        return next_deltas

    def print_weights(self):
        for layer in self.layers:
            if layer.location is not Layer.Location.INPUT:
                print(layer.location)
                print(layer.weights)

    def learn(self, inp, target):
        prediction = self.predict(inp, True)
        delta = self.__comp_delta(prediction, target)
        self.back_propagate(delta)
        if log.getLogger().getEffectiveLevel() == log.DEBUG:
            self.print_weights()

    # comp delta = prediction - expected value
    # result size = last layer node count
    def __comp_delta(self, prediction, target):
        # todo: function approach
        delta = subtract_vectors(prediction, target)
        log.debug(f"Delta: {delta}")
        return delta
