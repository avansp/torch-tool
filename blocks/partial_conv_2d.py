###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

# Modifications by Avan:
# 1. Removed multi-channel option
# 2. Mask is not returned. Instead it is stored under self.mask_out variable.
#    Return value from forward() is always an image.
# 3. Create additional padding that the padding will be learned by partial conv
#    So the output size will be the same if stride==1.
# 4. Mask input argument is mandatory
# 5. Removed last_size
# 6. Mask should have the same channel as the image channel.
#    If mask channel is 1, then the mask will be replicated along the channel axes.

import torch
import torch.nn.functional as F
from torch import nn, cuda


class PartialConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_mask_updater = None
        self.slide_winsize = None
        self.mask_out = None
        self.mask_ratio = None

        # calculate new padding to learn
        # see also torch.nn.functional.pad() function
        new_pad = tuple(int((s - 1) / 2) for s in self.kernel_size)
        self.pconv_padding = (new_pad[0], new_pad[0], new_pad[1], new_pad[1])

        # default mask is None (no hole), but we don't know yet its size
        self.mask_in = None

    def set_mask(self, mask_in):
        # check valid sizes
        mask_shape = mask_in.shape
        assert len(mask_shape) == 4, "Mask shape must be: [BATCH, 1, HEIGHT, WIDTH]"
        assert mask_shape[1] == 1, "Mask channel must be 1"
        self.mask_in = mask_in

    def forward(self, image_in):

        # check the validity of the image input
        im_shape = image_in.shape
        assert len(im_shape) == 4, "Image shape must be: [BATCH, CHANNELS, HEIGHT, WIDTH]"

        # get the mask, create one if it's still None
        if self.mask_in is None:
            self.mask_in = torch.ones((im_shape[0], 1, im_shape[2], im_shape[3]), device=self.weight.device)

        # check equal shape between image & mask
        mask_shape = self.mask_in.shape
        assert im_shape[0] == mask_shape[0] and im_shape[2] == mask_shape[2] and im_shape[3] == mask_shape[3], \
            "Image and mask sizes do not match"

        # if mask channel is 1 but image channel > 1, then we'll repeat the mask channel
        if mask_shape[1] == 1 and self.in_channels > 1:
            self.mask_in = self.mask_in.repeat(1, self.in_channels, 1, 1)

        # pad image_in & mask_in, which will be part of the learning
        # note that mask==0 is the regions to learn for the inpainting
        image_in = F.pad(image_in, self.pconv_padding, value=0).to(self.weight.device)
        mask_in = F.pad(self.mask_in, self.pconv_padding, value=0).to(self.weight.device)

        # prepare weight for the mask
        if self.weight_mask_updater is None:
            self.weight_mask_updater = torch.ones(self.out_channels,
                                                  self.in_channels,
                                                  self.kernel_size[0],
                                                  self.kernel_size[1]).to(mask_in)
            self.slide_winsize = self.weight_mask_updater.shape[1] * \
                                 self.weight_mask_updater.shape[2] * self.weight_mask_updater.shape[3]

        with torch.no_grad():
            self.mask_out = F.conv2d(mask_in,
                                     self.weight_mask_updater, bias=None,
                                     stride=self.stride, padding=self.padding,
                                     dilation=self.dilation, groups=1)

            # for mixed precision training, change 1e-8 to 1e-6
            self.mask_ratio = self.slide_winsize / (self.mask_out + 1e-8)
            self.mask_out = torch.clamp(self.mask_out, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.mask_out)

        raw_out = super(PartialConv2d, self).forward(torch.mul(image_in, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            image_out = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            image_out = torch.mul(image_out, self.mask_out)
        else:
            image_out = torch.mul(raw_out, self.mask_ratio)

        return image_out
