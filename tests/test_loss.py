from unittest import TestCase
from loss import *


class Test(TestCase):
    def test_mse_vector(self):
        target = [2, 2, 2, 0]
        actual = [2, 6, 3, -4]
        expected = [0, 8, 0.5, 8]

        actual = mse(target, actual, LossType.FUNCTION)

        self.assertEqual(expected, actual)

    def test_mse_vector_derivative(self):
        target = [2, 2, 2, 0]
        actual = [2, 6, 3, -4]
        expected = [0, 4, 1, -4]

        actual = mse(target, actual, LossType.DERIVATIVE)

        self.assertEqual(expected, actual)