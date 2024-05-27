from unittest import TestCase
from activation import *
from network import Network


class Test(TestCase):
    def test_relu_vector(self):
        vector = [-5, 7, 0, -2, 1]
        expected = [0, 7, 0, 0, 1]

        actual = relu(vector, ActivationType.FUNCTION)

        self.assertEqual(expected, actual)

    def test_relu_vector_derivative(self):
        vector = [-5, 7, 0, -2, 1, 7, 0.2]
        expected = [0, 1, 0, 0, 1, 1, 1]

        actual = relu(vector, ActivationType.DERIVATIVE)

        self.assertEqual(expected, actual)

    def test_sigmoid_vector(self):
        vector = [-1, 0, 2]
        expected = [0.26894, 0.5, 0.88080]

        actual = sig(vector, ActivationType.FUNCTION)

        self.assertEqual(expected, round_vector(actual, 5))

    def test_sigmoid_vector_derivative(self):
        vector = [-1, 0, 2]
        expected = [0.19661, 0.25, 0.10499]

        actual = sig(vector, ActivationType.DERIVATIVE)

        self.assertEqual(expected, round_vector(actual, 5))

    def test_tanh_vector(self):
        vector = [-1, 0, 2]
        expected = [-0.76159, 0, 0.96403]

        actual = tanh(vector, ActivationType.FUNCTION)

        self.assertEqual(expected, round_vector(actual, 5))

    def test_tanh_vector_derivative(self):
        vector = [-1, 0, 2]
        expected = [0.41997, 1, 0.07065]

        actual = tanh(vector, ActivationType.DERIVATIVE)

        self.assertEqual(expected, round_vector(actual, 5))

    def test_activation_setting(self):
        activation = [relu, sig, tanh, none]
        network_activation = Network([3, 3, 3, 3, 3], None, activation)
        activations_expected = [none, relu, sig, tanh, none]
        activations_actual = []

        for layer in network_activation.layers:
            activations_actual.append(layer.activation)

        network_no_activation = Network([3, 3, 3, 3, 3])
        no_activations_expected = [none, none, none, none, none]
        no_activations_actual = []

        for layer in network_no_activation.layers:
            no_activations_actual.append(layer.activation)

        self.assertEqual(activations_expected, activations_actual)
        self.assertEqual(no_activations_expected, no_activations_actual)


def round_vector(vector, precision):
    for i in range(len(vector)):
        vector[i] = round(vector[i], precision)
    return vector
