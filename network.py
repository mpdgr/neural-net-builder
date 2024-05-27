import logging as log

from activation import *
from layer import Layer
from loss import *


class Network:
    """
    Builds a dense neural network.

    To create a network define:
    - nodes list: list size determines nr of layers and list[i] determines nr of parameters at given layer
      ex. nodes = [50, 100, 10] for network of 3 layers of respectively 50, 100 and 10 neurons
    - dropout list: defines dropout rate at each layer except for output layer. Dropout layer size must equal
      nodes list size - 1 (no dropout on output layer). If no param is used, no dropout is applied.
    - activations list: defines activation function used at each layer except for input layer. Activations list size
      must equal nodes list size - 1 (no activation on input layer). If no param is used, no activation is applied.

    Example:
        network = Network([50, 100, 10], [0, 0.3], [sig, sig])
        creates 3-layer network with input size of 50, middle layer of 100 neurons, output size 10,
        0.3 dropout rate at middle layer and sigmoid activation on middle and output layers

    To train the network feed the learn function with input/target list pairs. Input list size must match nr of
    parameters in input layer of the network and target list size must match nr of parameters in network output
    layer.

    To use trained network, apply predict function to the input - input size must match nr of parameters in
    network input layer.

    By default, network uses mean square error as loss function and 0.1 alpha learning rate.
    """

    # layers list
    layers = []

    # dropout list -> dropout rate at each layer
    dropout = []

    # activation list
    activation = []

    # loss function -> default: mse
    loss = mse

    layers_count = None

    def __init__(self, nodes, dropout=None, activation=None):
        layers_count = len(nodes)
        if layers_count < 2:
            raise InvalidArgumentException("Network must include at lease two layers")

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
                raise InvalidArgumentException("Dropout list length must equal layers list - 1")
            else:
                for i in range(layers_count - 1):
                    layers[i].dropout_rate = dropout[i]

        # set activation functions
        if activation is not None:
            if len(activation) != len(layers) - 1:
                raise InvalidArgumentException("Activation list length must equal layers list - 1")
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

        # mse - default loss function
        self.loss = mse

        layers_info = [f"[{layer.input_count}, {layer.node_count}]" for layer in self.layers]
        log.info(f"Initialized network with layers: {', '.join(layers_info)}")

    # for inference phase set training to false
    def predict(self, inp, training=False):
        prediction = inp
        for i in range(0, self.layers_count):
            prediction = self.layers[i].forward_pass(prediction, training)
        log.debug(f'Input: {inp} -> Prediction: {prediction}')
        return prediction

    def learn(self, inp, target):
        prediction = self.predict(inp, True)
        loss_derivative = self.loss(target, prediction, LossType.DERIVATIVE)
        self.__back_propagate(loss_derivative)

    def __back_propagate(self, gradient):
        next_gradient = gradient
        for i in range(0, self.layers_count - 1):
            next_gradient = self.layers[self.layers_count - i - 1].backward_pass(next_gradient)
        return next_gradient

    # for debugging
    def print_weights(self):
        for layer in self.layers:
            if layer.location is not Layer.Location.INPUT:
                print(layer.location)
                print(layer.weights)
