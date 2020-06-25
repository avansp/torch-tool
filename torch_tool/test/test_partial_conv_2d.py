import unittest
from torch_tool.pconv import PartialConv2d
import torch


class PartialConv2dUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_default_pconv2d(self):
        """
        Purpose: to check outputs are valid given a default object creation.
        """

        p = PartialConv2d(1, 16, 3).to(self.device)  # input=1, output=16, kernel_size=3
        x = torch.rand(5, 1, 32, 32)                 # input shape = 5 x 1 x 32 x 32, mask == none

        y = p(x)                                     # output shape should be = 5 x 16 x 32 x 32
        self.assertEqual(y.shape, torch.Size([5, 16, 32, 32]))

        # since the mask is none (no hole), then the mask is not updated
        self.assertTrue(p.weight_mask_updater.equal(torch.ones(16, 1, 3, 3, device=self.device)))
        self.assertTrue(p.mask_out.equal(torch.ones(5, 16, 32, 32, device=self.device)))

        # check slide win size
        self.assertEqual(p.slide_winsize, 9)  # winsize = kernel_size[0] * kernel_size[1] * num_channels

        # since there is no mask, then the mask_out will be all 1.0
        self.assertTrue(p.mask_out.equal(torch.ones(5, 16, 32, 32, device=self.device)))

        # check the mask ratio
        # - for the kernel_size == 3, padding is 1
        # - for the image_size of 32x32, the non-padding area for mask_ratio will 1.0
        self.assertTrue(p.mask_ratio[:, :, 1:-1, 1:-1].equal(torch.ones(5, 16, 30, 30, device=self.device)))
        z1 = 1.5 * torch.ones(5, 16, 30, device=self.device)
        self.assertTrue(p.mask_ratio[:, :, 0, 1:-1].equal(z1))
        self.assertTrue(p.mask_ratio[:, :, -1, 1:-1].equal(z1))
        self.assertTrue(p.mask_ratio[:, :, 1:-1, 0].equal(z1))
        self.assertTrue(p.mask_ratio[:, :, 1:-1, -1].equal(z1))

        z2 = 2.25 * torch.ones(5, 16, device=self.device)
        self.assertTrue(p.mask_ratio[:, :, 0, 0].equal(z2))
        self.assertTrue(p.mask_ratio[:, :, 0, -1].equal(z2))
        self.assertTrue(p.mask_ratio[:, :, -1, 0].equal(z2))
        self.assertTrue(p.mask_ratio[:, :, -1, -1].equal(z2))

