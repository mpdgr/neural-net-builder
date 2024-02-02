from unittest import TestCase
from activation import *


class Test(TestCase):
    def test_relu_vector(self):
        vector = [-5, 7, 0, -2, 1]
        expected = [0, 7, 0, 0, 1]

        actual = __relu_vector(vector)

        self.assertEqual(expected, actual)

