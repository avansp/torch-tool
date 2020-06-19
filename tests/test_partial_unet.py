import unittest
from blocks import PartialEncoderBlock, PartialConv2d
import torch
from torch.nn import ReLU, BatchNorm2d


class PartialEncoderBlockUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_default_partial_encoder(self):
        pe = PartialEncoderBlock(3, 64, 3).to(self.device)

        self.assertEqual(len(pe.model), 3)
        self.assertEqual(type(pe.model[0]), PartialConv2d)
        self.assertEqual(type(pe.model[1]), BatchNorm2d)
        self.assertEqual(type(pe.model[2]), ReLU)

        self.assertEqual(pe.model[0].in_channels, 3)
        self.assertEqual(pe.model[0].out_channels, 64)

        # the default with reduce the spatial dimension into half
        z = torch.zeros(2, 3, 150, 150, device=self.device)
        self.assertEqual(pe(z).shape, (2, 64, 75, 75))

    def test_partial_encoder_no_batchnorm(self):
        pe = PartialEncoderBlock(3, 64, 3, bn=False).to(self.device)

        self.assertEqual(len(pe.model), 2)
        self.assertEqual(type(pe.model[0]), PartialConv2d)
        self.assertEqual(type(pe.model[1]), ReLU)

