import torch
from torch import nn
from . import PartialConv2d

__all__ = ["PartialDecoderBlock"]

class PartialDecoderBlock(nn.Module):
    """
    ==> upsampling ==> concat ==> pconv

    Notes:
    * upsampling uses interpolation (nearest neighbour)
    * upsampling is down individually: upsampling_img, upsampling_mask
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), bn=True, relu_factor=0.0):
        super(PartialDecoderBlock, self).__init__()

        self.pconv_block = nn.ModuleList()
        self.pconv_block.append(PartialConv2d(in_channel + out_channel, out_channel, kernel_size, stride=stride))
        if bn:
            self.pconv_block.append(nn.BatchNorm2d(out_channel))
        self.pconv_block.append(nn.LeakyReLU(relu_factor, inplace=True))

        self.__in_channels = in_channel

        # for concatenation - and it should be set carefully
        self.__concat_image = None
        self.__concat_mask = None

    def get_mask_out(self):
        return self.pconv_block[0].mask_out

    def set_concat_image(self, concat_image, clone=False):
        assert len(concat_image.shape) == 4, "Image must be in (batch, channels, height, width)."
        assert concat_image.shape[1] + self.__in_channels == self.pconv_block[0].in_channels, "Invalid channels"
        if clone:
            self.__concat_image = concat_image.clone().detach()
        else:
            self.__concat_image = concat_image

    def set_concat_mask(self, concat_mask, clone=False):
        assert self.__concat_image is not None, "Set concatenate image first, before mask to validate their shape."
        assert concat_mask.shape == self.__concat_image.shape, "The shape of the concatenate mask does not match with the concatenate image."
        if clone:
            self.__concat_mask = concat_mask.clone().detach()
        else:
            self.__concat_mask = concat_mask

    def set_concat(self, concat_image, concat_mask, clone=False):
        self.set_concat_image(concat_image, clone=clone)
        self.set_concat_mask(concat_mask, clone=clone)

    def forward(self, x):
        """
        Forward passing call, inherited.

        :param x can be:
          - an image tensor --> mask is None
          - a tuple of (image, mask) tensors

        :return: a tensor
        """

        # check if the concatenated image & mask
        assert self.__concat_image is not None, "Must set the concatenated image before calling this forward function."
        if self.__concat_mask is None:
            self.__concat_mask = torch.ones_like(self.__concat_image)

        # since the partial convolution is performed at the end of the block, we need to process both
        # the input image & mask separately
        if type(x) is tuple:
            assert len(x) == 2, "Input tuple must be (image, mask)"

            image_in = x[0]
            mask_in = x[1]

            assert image_in.shape == mask_in.shape, "Image and mask shapes do not match"
        else:
            image_in = x
            mask_in = torch.ones_like(image_in)

        # to interpolate to the same size of the encoder output
        spatial_size_out = self.__concat_image.shape[2:]

        # interpolate separately
        image_in = nn.functional.interpolate(image_in, size=spatial_size_out)
        mask_in = nn.functional.interpolate(mask_in, size=spatial_size_out)

        # concat separately
        image_out = torch.cat((image_in, self.__concat_image), 1)
        mask_out = torch.cat((mask_in, self.__concat_mask), 1)

        # perform the partial convolution
        y = self.pconv_block[0]((image_out, mask_out))
        for m in self.pconv_block[1:]:
            y = m(y)

        # output
        return y
