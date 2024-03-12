from unittest import TestCase
from activation import *


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



def round_vector(vector, precision):
    for i in range(len(vector)):
        vector[i] = round(vector[i], precision)
    return vector