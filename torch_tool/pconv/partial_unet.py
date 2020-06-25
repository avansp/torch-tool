from torch import nn
from . import PartialEncoderBlock, PartialDecoderBlock

__all__ = ["PartialUNet"]


class PartialUNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(PartialUNet, self).__init__()

        # see Table 2 on https://arxiv.org/pdf/1804.07723.pdf

        # ENCODERS
        self.pconv1 = PartialEncoderBlock(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), bn=False, relu_factor=0.0)
        self.pconv2 = PartialEncoderBlock(64, 128, kernel_size=(5 ,5), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv3 = PartialEncoderBlock(128, 256, kernel_size=(5 ,5), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv4 = PartialEncoderBlock(256, 512, kernel_size=(3 ,3), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv5 = PartialEncoderBlock(512, 512, kernel_size=(3 ,3), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv6 = PartialEncoderBlock(512, 512, kernel_size=(3 ,3), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv7 = PartialEncoderBlock(512, 512, kernel_size=(3 ,3), stride=(2 ,2), bn=True, relu_factor=0.0)
        self.pconv8 = PartialEncoderBlock(512, 512, kernel_size=(3 ,3), stride=(2 ,2), bn=True, relu_factor=0.0)

        # DECODERS
        self.pconv9 = PartialDecoderBlock(512, 512, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv10 = PartialDecoderBlock(512, 512, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv11 = PartialDecoderBlock(512, 512, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv12 = PartialDecoderBlock(512, 512, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv13 = PartialDecoderBlock(512, 256, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv14 = PartialDecoderBlock(256, 128, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv15 = PartialDecoderBlock(128, 64, kernel_size=(3 ,3), bn=True, relu_factor=0.2)
        self.pconv16 = PartialDecoderBlock(64, out_channel, kernel_size=(3 ,3), bn=False, relu_factor=0.0)

    def get_mask_out(self):
        return self.pconv16.get_mask_out()

    def forward(self, img):

        # first convolution
        y1 = self.pconv1(img)

        # next ones
        y2 = self.pconv2((y1, self.pconv1.get_mask_out()))
        y3 = self.pconv3((y2, self.pconv2.get_mask_out()))
        y4 = self.pconv4((y3, self.pconv3.get_mask_out()))
        y5 = self.pconv5((y4, self.pconv4.get_mask_out()))
        y6 = self.pconv6((y5, self.pconv5.get_mask_out()))
        y7 = self.pconv7((y6, self.pconv6.get_mask_out()))
        y8 = self.pconv8((y7, self.pconv7.get_mask_out()))

        # decoding
        self.pconv9.set_concat(y7, self.pconv7.get_mask_out())
        y9 = self.pconv9((y8, self.pconv8.get_mask_out()))

        self.pconv10.set_concat(y6, self.pconv6.get_mask_out())
        y10 = self.pconv10((y7, self.pconv7.get_mask_out()))

        self.pconv11.set_concat(y5, self.pconv5.get_mask_out())
        y11 = self.pconv11((y6, self.pconv6.get_mask_out()))

        self.pconv12.set_concat(y4, self.pconv4.get_mask_out())
        y12 = self.pconv12((y5, self.pconv5.get_mask_out()))

        self.pconv13.set_concat(y3, self.pconv3.get_mask_out())
        y13 = self.pconv13((y4, self.pconv4.get_mask_out()))

        self.pconv14.set_concat(y2, self.pconv2.get_mask_out())
        y14 = self.pconv14((y3, self.pconv3.get_mask_out()))

        self.pconv15.set_concat(y1, self.pconv1.get_mask_out())
        y15 = self.pconv15((y2, self.pconv2.get_mask_out()))

        self.pconv16.set_concat(img[0], img[1])
        y16 = self.pconv16((y1, self.pconv1.get_mask_out()))

        return y16
