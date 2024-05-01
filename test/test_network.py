import logging
from unittest import TestCase

from activation import none, relu, sig
from network import Network


class TestNetwork(TestCase):

    def test_learn_2x2x1(self):
        layers = [2, 2, 1]

        inp = [2, 3]
        target = [1]

        weights1 = [
            [0.11, 0.12],
            [0.21, 0.08]
        ]

        weights2 = [
            [0.14],
            [0.15]
        ]

        network = Network(layers, None, None, True)
        network.layers[1].weights = weights1
        network.layers[2].weights = weights2

        network = Network(layers, None, None, True)
        network.layers[1].weights = weights1
        network.layers[2].weights = weights2

        print(f"Start weights: ")
        network.print_weights()
        network.learn(inp, target)
        print(f"End weights: ")
        network.print_weights()

        network.learn(inp, target)


    def test_learn_2x2x2(self):
        layers = [2, 2, 2]

        inp = [1, 2]
        target = [25.5, 30]

        weights1 = [
            [0.5, 1],
            [1.5, 2]
        ]

        weights2 = [
            [2.5, 3],
            [3.5, 4]
        ]

        network = Network(layers, None, None, True)
        network.layers[1].weights = weights1
        network.layers[2].weights = weights2

        print(f"Start weights: ")
        network.print_weights()
        network.learn(inp, target)
        print(f"End weights: ")
        network.print_weights()

        weights1_expected = [
            [0.1625, 0.5375],
            [0.825, 1.075]
        ]

        weights2_expected = [
            [2.2375, 2.825],
            [3.125, 3.75]
        ]

        weights1_actual = network.layers[1].weights
        weights2_actual = network.layers[2].weights

        # round matrices for float rounding error
        round_matrix(weights1_actual, 10)
        round_matrix(weights2_actual, 10)

        print('act:')
        for i in network.layers:
            print(i.activation)

        self.assertEqual(weights1_expected, weights1_actual)
        self.assertEqual(weights2_expected, weights2_actual)

    def test_learn_3x2(self):
        layers = [3, 2]

        inp = [1, 2, 3]
        target = [20, 15]

        weights = [
            [1, 1.5],
            [2, 2.5],
            [3, 3.5]
        ]

        network = Network(layers, None, None, True)
        network.layers[1].weights = weights

        print(f"Start weights: ")
        network.print_weights()
        network.learn(inp, target)
        print(f"End weights: ")
        network.print_weights()

        weights_expected = [
            [1.6, 1.3],
            [3.2, 2.1],
            [4.8, 2.9]
        ]

        weights_actual = network.layers[1].weights

        # round matrices for float rounding error
        round_matrix(weights_actual, 10)

        self.assertEqual(weights_expected, weights_actual)

    def test_learn_3x1(self):
        layers = [2, 3]

        inp = [1, 2, 3]
        target = [20]

        weights = [
            [1],
            [2],
            [3]
        ]

        expected = [1.6, 3.2, 4.8]

        network = Network(layers, None, None, True)
        network.layers[1].weights = weights

        print(f"Start weights: ")
        network.print_weights()
        network.learn(inp, target)
        print(f"End weights: ")
        network.print_weights()

        weights_expected = [
            [1.6],
            [3.2],
            [4.8]
        ]

        weights_actual = network.layers[1].weights

        # round matrices for float rounding error
        round_matrix(weights_actual, 10)

        self.assertEqual(weights_expected, weights_actual)

    def test_learn(self):
        layers = [100, 400, 800, 600, 200, 400, 300, 250, 1000, 25]

        inp = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ]
        target = [
            20, 40, 60, 80, 100,
            20, 40, 60, 80, 100,
            20, 40, 60, 80, 100,
            20, 40, 60, -80, 100,
            20, 40, 60, 80, 100
        ]

        network = Network(layers, None, None, True)

        network.learn(inp, target)

    def test_3x4x1(self):
        # ok: 300 iteracji/ layers = [3, 4, 4, 1]
        # ok: 500 iteracji/ layers = [3, 4, 2, 1]
        # dziala na relu i tanh

        layers = [3, 4, 2, 1]

        inputs = [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ]

        outputs = [[1], [1], [0], [0]]

        network = Network(layers, None, [relu, relu, none], True)

        for iteration in range(300):
            for i in range(0, len(inputs)):
                network.learn(inputs[i], outputs[i])

        for i in range(0, len(inputs)):
            print(f"\n----->Prediction for: {inputs[i]}; expected: {outputs[i]}")
            network.predict(inputs[i])


    def test_3x4x1_with_dropout(self):
        # w miarę ok: 300 iteracji -> wynik zbliżony lub same 0
        # layers = [3, 8, 2, 1]
        # dropout = [0, 0.3, 0]
        # dziala na relu
        logging.getLogger().setLevel(logging.DEBUG)

        layers = [3, 8, 2, 2, 1]
        dropout = [0, 0.3, 0, 0]

        inputs = [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ]

        outputs = [[1], [1], [0], [0]]

        network = Network(layers, dropout, [relu, relu, none, none], True)

        for iteration in range(300):
            for i in range(0, len(inputs)):
                network.learn(inputs[i], outputs[i])

        for i in range(0, len(inputs)):
            print(f"\n----->Prediction for: {inputs[i]}; expected: {outputs[i]}")
            network.predict(inputs[i])

    def test_dropout_set(self):
        layers = [3, 4, 2, 1]
        network = Network(layers, None, None, True)

        self.assertEqual(network.layers[0].dropout_rate, 0)

        layers = [3, 4, 2, 1]
        network = Network(layers, [0.3, 0.4, 0.2], None, True)

        self.assertEqual(network.layers[0].dropout_rate, 0.3)
        self.assertEqual(network.layers[1].dropout_rate, 0.4)
        self.assertEqual(network.layers[2].dropout_rate, 0.2)
        self.assertEqual(network.layers[3].dropout_rate, 0)

        layers = [3, 4, 2, 1]
        network = Network(layers, [0.3, 0.4, 0.2], None, True)

        self.assertEqual(network.layers[0].dropout_rate, 0.3)
        self.assertEqual(network.layers[1].dropout_rate, 0.4)
        self.assertEqual(network.layers[2].dropout_rate, 0.2)
        self.assertEqual(network.layers[3].dropout_rate, 0)

        layers = [3, 4, 2, 1]
        network = Network(layers, [0.3, 0.4, 0.2, 0.3], True)


def round_matrix(matrix, precision):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = round(matrix[i][j], precision)
