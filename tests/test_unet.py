import unittest
import blocks
from models import VanillaUNet
import torch


class UNetUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_double_conv_block(self):
        block = blocks.DoubleConvBlock(3, 2).to(self.device)
        out_shape = block.get_output_shape(572, 572)
        self.assertEqual(out_shape, (568, 568))

    def test_down_sample_block(self):
        block = blocks.DownSampleBlock(3, 2).to(self.device)
        out_shape = block.get_output_shape(568, 568)
        self.assertEqual(out_shape, (280, 280))

    def test_vanilla_unet(self):
        vu = VanillaUNet(3, 2).to(self.device)
        out_shape = vu.get_output_shape(572, 572)
        self.assertEqual(out_shape, (388, 388))
