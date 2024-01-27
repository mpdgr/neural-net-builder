from unittest import TestCase
from network import Network


class TestNetwork(TestCase):

    def test_learn(self):
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

        network = Network(layers, True)
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

        self.assertEqual(weights1_expected, weights1_actual)
        self.assertEqual(weights2_expected, weights2_actual)


def round_matrix(matrix, precision):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = round(matrix[i][j], precision)
