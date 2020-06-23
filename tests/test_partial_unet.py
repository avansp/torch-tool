import unittest
import torch
from models import PartialUNet


class PartialUNetUnitTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_vanilla_unet(self):
        punet = PartialUNet(3, 3).to(self.device)

        im = torch.rand(2, 3, 512, 512, device=self.device)
        mask = torch.rand(2, 3, 512, 512, device=self.device)

        y = punet((im, mask))
        self.assertEqual(y.shape, (2, 3, 512, 512))
        self.assertEqual(punet.get_mask_out().shape, (2, 3, 512, 512))

