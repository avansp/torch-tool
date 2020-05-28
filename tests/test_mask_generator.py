import unittest
from utils import MaskGenerator
import numpy as np


class MaskGeneratorUnitTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_random_mask(self):
        mask_gen = MaskGenerator(100, 100, rand_seed=10)
        mask = mask_gen.sample()

        self.assertEqual((100, 100), mask.numpy().shape)
        np.testing.assert_array_equal(np.array([0, 1], dtype=np.uint8), np.unique(mask.numpy()))
