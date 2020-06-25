from torch import nn
from .unet_blocks import *


class VanillaUNet(nn.Module):
    """
    Encoding: DoubleConvEncoder + 4 * DownSample
    Decoding:
    """

    def __init__(self, in_channel=1, out_channel=2):
        """
        Strict from Rossenberger's UNet: input = 1 (greyscale), output = 2 (mask)
        """
        super(VanillaUNet, self).__init__()

        self.layer_names = ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5']
        self.input_channel = in_channel
        self.output_channel = out_channel

        self.encoders = nn.ModuleDict([
            (name, block(ic, oc)) for name, block, ic, oc in zip(
                self.layer_names,
                [DoubleConvBlock, DownSampleBlock, DownSampleBlock,
                 DownSampleBlock, DownSampleBlock],
                [in_channel, 64, 128, 256, 512],
                [64, 128, 256, 512, 1024]
            )
        ])

        self.decoders = nn.ModuleDict([
            (name, UpSampleBlock(ic)) for name, ic in zip(
                self.layer_names,
                [128, 256, 512, 1024]
            )
        ])

        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, in_image):
        out_image = in_image

        # encoding
        for k in self.layer_names:
            out_image = self.encoders[k](out_image)

        # decoding (from the back)
        for k in list(self.decoders.keys())[len(self.decoders)::-1]:
            out_image = self.decoders[k](out_image, self.encoders[k].get_output())

        # final layer
        out_image = self.output_layer(out_image)

        return out_image

    def get_output_shape(self, height, width):
        Z = torch.zeros(1, self.input_channel, height, width, device=next(self.parameters()).device)
        out = self.forward(Z)
        return tuple(out.shape[-2:])
