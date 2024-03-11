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

