import unittest
from torch_tool.pconv import PartialEncoderBlock, PartialConv2d, PartialDecoderBlock
import torch
from torch.nn import LeakyReLU, BatchNorm2d


class PartialEncoderBlockUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_default_partial_encoder(self):
        pe = PartialEncoderBlock(3, 64, 3).to(self.device)

        self.assertEqual(len(pe.pconv_block), 3)
        self.assertEqual(type(pe.pconv_block[0]), PartialConv2d)
        self.assertEqual(type(pe.pconv_block[1]), BatchNorm2d)
        self.assertEqual(type(pe.pconv_block[2]), LeakyReLU)

        self.assertEqual(pe.pconv_block[0].in_channels, 3)
        self.assertEqual(pe.pconv_block[0].out_channels, 64)

        # the default with reduce the spatial dimension into half
        z = torch.zeros(2, 3, 150, 150, device=self.device)
        self.assertEqual(pe(z).shape, (2, 64, 75, 75))

    def test_partial_encoder_no_batchnorm(self):
        pe = PartialEncoderBlock(3, 64, 3, bn=False).to(self.device)

        self.assertEqual(len(pe.pconv_block), 2)
        self.assertEqual(type(pe.pconv_block[0]), PartialConv2d)
        self.assertEqual(type(pe.pconv_block[1]), LeakyReLU)


class PartialDecoderBlockUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_default_partial_decoder(self):
        pd = PartialDecoderBlock(512, 512, 3).to(self.device)

        self.assertEqual(len(pd.pconv_block), 3)
        self.assertEqual(type(pd.pconv_block[0]), PartialConv2d)
        self.assertEqual(type(pd.pconv_block[1]), BatchNorm2d)
        self.assertEqual(type(pd.pconv_block[2]), LeakyReLU)

        im = torch.rand(5, 512, 16, 16, device=self.device)
        pd.set_concat_image(im)
        z = torch.rand(5, 512, 8, 8, device=self.device)
        y = pd(z)

        self.assertEqual(y.shape, (5, 512, 16, 16))

        m_out = pd.get_mask_out()
        self.assertEqual(m_out.shape, (5, 512, 16, 16))
        self.assertTrue(torch.ones_like(y).equal(m_out))
