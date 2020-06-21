import torch.nn as nn
from blocks import PartialConv2d


class PartialEncoderBlock(nn.Module):
    """
    ==> pconv ==> [bn] ==> relu

    pconv with stride=2 to reduce the size by half
    """

    def __init__(self, in_channel, out_channel, kernel_size, bn=True, stride=(2,2)):
        super(PartialEncoderBlock, self).__init__()

        self.pconv_block = nn.ModuleList()
        self.pconv_block.append(PartialConv2d(in_channel, out_channel, kernel_size, stride=stride))
        if bn:
            self.pconv_block.append(nn.BatchNorm2d(out_channel))
        self.pconv_block.append(nn.ReLU(True))

    def get_mask_out(self):
        return self.pconv_block[0].mask_out

    def forward(self, x):
        """
        x can be:
          - an image tensor --> mask is None
          - a tuple of (image, mask) tensors
        """
        y = self.pconv_block[0](x)
        for m in self.pconv_block[1:]:
            y = m(y)

        # return image output
        return y