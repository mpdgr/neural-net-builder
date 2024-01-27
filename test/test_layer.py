from unittest import TestCase

from layer import Layer


class TestLayer(TestCase):
    def test_learning_round(self):
        inp = [1, 2, 3]
        target = [20, 15]

        weights = [
            [1, 1.5],
            [2, 2.5],
            [3, 3.5]
        ]

        expected = [-3, -7, -11]

        layer = Layer(2, 3, Layer.Location.HIDDEN, weights, True)
        actual = layer.output_round(inp, target)

        self.assertEqual(expected, actual)

    def test_learning_round_one_out(self):
        inp = [1, 2, 3]
        target = [20]

        weights = [
            [1],
            [2],
            [3]
        ]

        expected = [-3, -7, -11]

        layer = Layer(1, 3, Layer.Location.HIDDEN, weights, True)
        actual = layer.output_round(inp, target)

        self.assertEqual(expected, actual)

