from torch import nn
import torch
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """
    ==> conv ==> BN ==> relu ==> conv ==> BN ==> relu ==>
    in_channels --> out_channels --> out_channels

    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1):
        super(DoubleConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out_image = None

    def forward(self, in_image):
        self.out_image = self.model(in_image)
        return self.out_image

    def get_output(self):
        return self.out_image


class DownSampleBlock(nn.Module):
    """
    (in_channel) ==> MaxPool
                        |
                        ==> DoubleConvBlock ==> (out_channel)
    """

    def __init__(self, in_channel, out_channel):
        super(DownSampleBlock, self).__init__()

        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvBlock(in_channel, out_channel)
        )

    def forward(self, in_image):
        return self.model(in_image)

    def get_output(self):
        return self.model[1].get_output()


class UpSampleBlock(nn.Module):
    """
    up-sampling ==> concat ==> double-convolution
    """

    def __init__(self, in_channel):
        super(UpSampleBlock, self).__init__()

        self.skip_image = None
        self.up_sample = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channel, in_channel // 2)

    def forward(self, in_image, skip_image):
        # up sample the input image
        out_image = self.up_sample(in_image)

        # skip image has additional pad size due to ConvTranspose2d output
        pad_size = [
            item for sublist in
            torch.tensor(out_image.shape[-2:]) - torch.tensor(skip_image.shape[-2:])
            for item in [sublist.item() // 2] * 2]

        # the concatenation
        out_image = torch.cat((F.pad(skip_image, pad_size), out_image), dim=1)

        # pass through the convolution
        out_image = self.double_conv(out_image)

        return out_image
