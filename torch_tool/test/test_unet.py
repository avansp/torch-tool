import torch
import unittest
from torch_tool.unet import *


class UNetUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_double_conv_block(self):
        block = DoubleConvBlock(3, 2).to(self.device)
        x = torch.zeros(2, 3, 572, 572, device=self.device)
        z = block(x)
        self.assertEqual(z.shape, (2, 2, 568, 568))

    def test_down_sample_block(self):
        block = DownSampleBlock(3, 2).to(self.device)
        x = torch.zeros(2, 3, 568, 568, device=self.device)
        z = block(x)
        self.assertEqual(z.shape, (2, 2, 280, 280))

    def test_vanilla_unet(self):
        vu = VanillaUNet(3, 2).to(self.device)
        x = torch.zeros(2, 3, 572, 572, device=self.device)
        z = vu(x)
        self.assertEqual(z.shape, (2, 2, 388, 388))
