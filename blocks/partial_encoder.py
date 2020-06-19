import torch.nn as nn
from blocks import PartialConv2d


class PartialEncoderBlock(nn.Module):
    """
    ==> pconv ==> [bn] ==> relu

    pconv with stride=2 to reduce the size by half
    """

    def __init__(self, in_channel, out_channel, kernel_size, bn=True, stride=(2,2)):
        super(PartialEncoderBlock, self).__init__()

        self.model = nn.ModuleList()
        self.model.append(PartialConv2d(in_channel, out_channel, kernel_size, stride=stride))
        if bn:
            self.model.append(nn.BatchNorm2d(out_channel))
        self.model.append(nn.ReLU(True))

    def get_mask_out(self):
        return self.model[0].mask_out

    def forward(self, x):
        """
        x can be:
          - an image tensor --> mask is None
          - a tuple of (image, mask) tensors
        """
        y = self.model[0](x)
        for m in self.model[1:]:
            y = m(y)

        # return image output
        return y