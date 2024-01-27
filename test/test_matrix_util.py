from unittest import TestCase
from util import matrix_util


class TestMatrixUtil(TestCase):

    def test_matrix_product_square(self):
        m1 = [[2, 5], [1, -2]]
        m2 = [[3, -1], [7, 4]]
        expected = [[41, 18], [-11, -9]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_not_square(self):
        m1 = [[1, 0, 2], [-1, 3, 1]]
        m2 = [[3, 1], [2, 1], [1, 0]]
        expected = [[5, 1], [4, 2]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_invalid(self):
        m1 = [[2, 5, 6]]
        m2 = [[3, -1], [7, 4]]
        expected = None

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_first_vector(self):
        m1 = [[2, 5]]
        m2 = [[3, -1], [7, 4]]
        expected = [[41, 18]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_first_vector_unwrapped(self):
        m1 = [2, 5]
        m2 = [[3, -1], [7, 4]]
        expected = [[41, 18]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_second_vector(self):
        m1 = [[1, 2, 3, 4]]
        m2 = [[5], [6], [7], [8]]
        expected = [[70]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_matrix_product_larger(self):
        m1 = [[5, -1, 0], [4, 9, 4], [-10, 0, 7], [1, 2, 3]]
        m2 = [[1, -5, 5], [6, -2, 1], [2, 13, -3]]
        expected = [[-1, -23, 24], [66, 14, 17], [4, 141, -71], [19, 30, -2]]

        actual = matrix_util.matrix_product(m1, m2)

        self.assertEqual(expected, actual)

    def test_extract_column_first_col(self):
        m = [[1, 0, 2], [-1, 3, 1]]
        expected = [1, -1]

        actual = matrix_util.extract_column(m, 0)

        self.assertEqual(expected, actual)

    def test_extract_column_last_col(self):
        m = [[3, 1], [2, 1], [1, 0]]
        expected = [1, 1, 0]

        actual = matrix_util.extract_column(m, 1)

        self.assertEqual(expected, actual)

    def test_vector_dot_product(self):
        v1 = [1, 2, 3]
        v2 = [3, 2, 1]
        expected = 10

        actual = matrix_util.vector_dot_product(v1, v2)

        self.assertEqual(expected, actual)

    def test_matrix_scalar_product(self):
        m1 = [[5, -1, 0], [4, 9, 4], [-10, 0, 7], [1, 2, 3]]
        scalar = 2
        expected = [[10, -2, 0], [8, 18, 8], [-20, 0, 14], [2, 4, 6]]

        actual = matrix_util.matrix_scalar_product(m1, scalar)

        self.assertEqual(expected, actual)

    def test_subtract_vectors(self):
        v1 = [1, 2, 3]
        v2 = [3, 2, 1]
        expected = [-2, 0, 2]

        actual = matrix_util.subtract_vectors(v1, v2)

        self.assertEqual(expected, actual)

    def test_subtract_vectors_with_unwrap(self):
        v1 = [[1, 2, 3]]
        v2 = [3, 2, 1]
        expected = [-2, 0, 2]

        actual = matrix_util.subtract_vectors(v1, v2)

        self.assertEqual(expected, actual)

    def test_vector_element_wise_product(self):
        v1 = [1, 2, 3]
        v2 = [3, 2, 1]
        expected = [3, 4, 3]

        actual = matrix_util.vector_element_wise_product(v1, v2)

        self.assertEqual(expected, actual)

    def test_vector_element_wise_product_with_unwrap_first(self):
        v1 = [[1, 2, 3]]
        v2 = [3, 2, 1]
        expected = [3, 4, 3]

        actual = matrix_util.vector_element_wise_product(v1, v2)

        self.assertEqual(expected, actual)

    def test_vector_element_wise_product_with_unwrap_second(self):
        v1 = [1, 2, 3]
        v2 = [[3, 2, 1]]
        expected = [3, 4, 3]

        actual = matrix_util.vector_element_wise_product(v1, v2)

        self.assertEqual(expected, actual)

    def test_subtract_vector_from_matrix_rows(self):
        vector = [1, 2, 3]
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [[0, 0, 0], [3, 3, 3], [6, 6, 6]]

        actual = matrix_util.subtract_vector_from_matrix_rows(vector, matrix)

        self.assertEqual(expected, actual)

    def test_subtract_vector_from_matrix_rows_with_unwrap(self):
        vector = [[1, 2, 3]]
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [[0, 0, 0], [3, 3, 3], [6, 6, 6]]

        actual = matrix_util.subtract_vector_from_matrix_rows(vector, matrix)

        self.assertEqual(expected, actual)

    def test_vector_matrix_row_wise_product(self):
        vector = [1, 2, 3]
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [[1, 4, 9], [4, 10, 18], [7, 16, 27]]

        actual = matrix_util.vector_matrix_row_wise_product(vector, matrix)

        self.assertEqual(expected, actual)

    def test_vector_matrix_row_wise_product_with_unwrap(self):
        vector = [[1, 2, 3]]
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [[1, 4, 9], [4, 10, 18], [7, 16, 27]]

        actual = matrix_util.vector_matrix_row_wise_product(vector, matrix)

        self.assertEqual(expected, actual)

    def test_vector_outer_product(self):
        v1 = [1, 2, 3]
        v2 = [4, 5]
        expected = [[4, 5], [8, 10], [12, 15]]

        actual = matrix_util.vector_outer_product(v1, v2)

        self.assertEqual(expected, actual)

    def test_subtract_matrices(self):
        m1 = [[5, -1, 0], [4, 9, 4], [-10, 0, 7], [1, 2, 3]]
        m2 = [[5, -1, 0], [4, 9, 4], [-10, 0, 7], [1, 2, 3]]
        expected = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        actual = matrix_util.subtract_matrices(m1, m2)

        self.assertEqual(expected, actual)

    def test_transpose_matrix(self):
        m1 = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected1 = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]

        actual1 = matrix_util.transpose_matrix(m1)

        m2 = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        expected2 = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]

        actual2 = matrix_util.transpose_matrix(m2)

        m3 = [
            [1, 2, 3]
        ]
        expected3 = [
            [1],
            [2],
            [3]
        ]

        actual3 = matrix_util.transpose_matrix(m3)

        m4 = [1, 2, 3]
        expected4 = [
            [1],
            [2],
            [3]
        ]

        actual4 = matrix_util.transpose_matrix(m4)

        self.assertEqual(expected1, actual1)
        self.assertEqual(expected2, actual2)
        self.assertEqual(expected3, actual3)
        self.assertEqual(expected4, actual4)

    def test_matrix_to_vector_row_major(self):
        m1 = [[5, -1, 0], [4, 9, 4], [-10, 0, 7], [1, 2, 3]]
        expected1 = [5, -1, 0, 4, 9, 4, -10, 0, 7, 1, 2, 3]

        actual1 = matrix_util.matrix_to_vector_row_major(m1)

        m2 = [[5], [4]]
        expected2 = [5, 4]

        actual2 = matrix_util.matrix_to_vector_row_major(m2)

        self.assertEqual(expected1, actual1)
        self.assertEqual(expected2, actual2)