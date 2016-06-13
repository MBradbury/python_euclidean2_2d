#!/usr/bin/env python

import unittest

import numpy as np

from euclidean import euclidean2_2d as euclidean

class CorrectnessTest(unittest.TestCase):

    def test_euclidean2_2d_same_int(self):

        a = np.array((1, 1))
        b = np.array((1, 1))

        self.assertEqual(euclidean(a, b), 0.0)

    def test_euclidean2_2d_same_float(self):

        a = np.array((1.0, 1.0))
        b = np.array((1.0, 1.0))

        self.assertEqual(euclidean(a, b), 0.0)

    def test_euclidean2_2d_difference_int(self):

        a = np.array((1, 1))
        b = np.array((2, 1))

        self.assertEqual(euclidean(a, b), 1.0)

    def test_euclidean2_2d_difference_float(self):

        a = np.array((1.0, 1.0))
        b = np.array((2.0, 1.0))

        self.assertEqual(euclidean(a, b), 1.0)

if __name__ == "__main__":
    unittest.main()
