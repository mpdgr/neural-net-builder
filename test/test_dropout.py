from unittest import TestCase
from dropout import apply_dropout, apply_gradient_dropout


class Test(TestCase):
    def test_apply_dropout_1(self):
        rate = 0.5
        v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        v2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        dropped_out_1 = apply_dropout(rate, v1)
        dropped_out_2 = apply_dropout(rate, v2)

        self.assertEqual(dropped_out_1[0].count(0), 10)
        self.assertEqual(dropped_out_2[0].count(0), 10)
        self.assertNotEqual(dropped_out_1, dropped_out_2)

    def test_apply_dropout_2(self):
        rate = 0.1
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        dropped_out = apply_dropout(rate, v)

        self.assertEqual(dropped_out[0].count(0), 2)

    def test_apply_dropout_3(self):
        rate = 0.5
        v = [1, 2, 3]

        dropped_out = apply_dropout(rate, v)

        self.assertEqual(dropped_out[0].count(0), 1)

    def test_apply_dropout_high(self):
        rate = 0.9
        v = [1, 2, 3]

        dropped_out = apply_dropout(rate, v)

        self.assertEqual(dropped_out[0].count(0), 0)

    def test_apply_dropout_none(self):
        rate = 0
        v = [1, 2, 3]

        dropped_out = apply_dropout(rate, v)

        self.assertEqual(dropped_out[0].count(0), 0)

    def test_apply_dropout_sum_of_elements_1(self):
        rate = 0.5
        v = [2, 2, 2]

        dropped_out = apply_dropout(rate, v)

        print(v)
        print(f'sum{sum(v)}')
        self.assertEqual(dropped_out[0].count(0), 1)
        self.assertTrue(- 0.0001 < sum(dropped_out[0]) - 6 < 0.0001)

    def test_apply_dropout_sum_of_elements_2(self):
        rate = 0.5
        v = [2, 2, 2, 2]

        dropped_out = apply_dropout(rate, v)

        print(v)
        print(f'sum{sum(v)}')
        self.assertEqual(dropped_out[0].count(0), 2)
        self.assertTrue(- 0.0001 < sum(dropped_out[0]) - 8 < 0.0001)

    def test_apply_dropout_sum_of_elements_3(self):
        rate = 0.1
        v = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        dropped_out = apply_dropout(rate, v)

        print(v)
        print(f'sum{sum(v)}')
        self.assertEqual(dropped_out[0].count(0), 1)
        self.assertTrue(- 0.0001 < sum(dropped_out[0]) - 10 < 0.0001)

    def test_apply_gradient_dropout(self):
        drop_indices = [1, 3, 5]
        v = [2, 4, 6, 8, 10, 12]
        expected = [4, 0, 12, 0, 20, 0]

        dropped_out = apply_gradient_dropout(drop_indices, v)

        print(dropped_out)
        print(f'sum{sum(v)}')
        self.assertEqual(expected, dropped_out)

    def test_apply_gradient_no_dropout(self):
        drop_indices = []
        v = [2, 4, 6, 8, 10, 12]
        expected = [2, 4, 6, 8, 10, 12]

        dropped_out = apply_gradient_dropout(drop_indices, v)

        print(dropped_out)
        print(f'sum{sum(v)}')
        self.assertEqual(expected, dropped_out)