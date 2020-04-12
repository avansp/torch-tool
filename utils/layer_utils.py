import torch


def fix_param(param):
    return param if type(param) is tuple else (param, param)


def conv2d_output_shape(height: int, width: int, conv2d: torch.nn.Conv2d):
    return tuple([
        round((sz + 2 * pad - dilation * (kernel - 1)) / stride) for (sz, pad, dilation, kernel, stride) in
        zip(
            (height, width),
            fix_param(conv2d.padding),
            fix_param(conv2d.dilation),
            fix_param(conv2d.kernel_size),
            fix_param(conv2d.stride)
        )
    ])


def maxpool2d_output_shape(height: int, width: int, maxpool2d: torch.nn.MaxPool2d):
    return tuple([
        round((sz + 2 * pad - dilation * (kernel - 1)) / stride) for (sz, pad, dilation, kernel, stride) in
        zip(
            (height, width),
            fix_param(maxpool2d.padding),
            fix_param(maxpool2d.dilation),
            fix_param(maxpool2d.kernel_size),
            fix_param(maxpool2d.stride)
        )
    ])


def convtranspose2d_output_shape(height: int, width: int, convt2d: torch.nn.ConvTranspose2d):
    return tuple([
        (sz - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        for (sz, stride, padding, dilation, kernel_size, output_padding) in
        zip(
            (height, width),
            fix_param(convt2d.stride),
            fix_param(convt2d.padding),
            fix_param(convt2d.dilation),
            fix_param(convt2d.kernel_size),
            fix_param(convt2d.output_padding)
        )
    ])
