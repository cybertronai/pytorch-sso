import torch
import torch.nn as nn
import torch.nn.functional as F


def test_im2col_conv2d(thr=1e-6):
    n = 2
    c_in, c_out, k = 3, 5, 3
    x = torch.randn(n, c_in, 28, 28)
    conv2d = nn.Conv2d(c_in, c_out, k, bias=False)
    output = conv2d(x)

    # im2col
    Mx = F.unfold(x, conv2d.kernel_size,
                  dilation=conv2d.dilation,
                  padding=conv2d.padding,
                  stride=conv2d.stride)

    # matmul as convolution
    W = conv2d.weight.view(conv2d.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


def test_im2col_conv3d(thr=1e-6):
    n = 2
    c_in, c_out, k = 4, 5, 3
    x = torch.randn(n, c_in, 28, 28, 28)
    conv3d = nn.Conv3d(c_in, c_out, 3, bias=False, padding=3)
    output = conv3d(x)

    # im2col
    kernel_size = conv3d.kernel_size
    stride = conv3d.stride
    assert conv3d.dilation == (1, 1, 1)
    pad_left = pad_right = conv3d.padding[0]
    pad_top = pad_bottom = conv3d.padding[1]
    pad_front = pad_back = conv3d.padding[2]
    x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])
    x_slices = x.unfold(
               2, kernel_size[0], stride[0]).unfold(
               3, kernel_size[1], stride[1]).unfold(
               4, kernel_size[2], stride[2])
    Mx = x_slices.flatten(start_dim=5)
    Mx = Mx.flatten(start_dim=2, end_dim=4)
    Mx = Mx.transpose(2, 3)
    Mx = Mx.flatten(start_dim=1, end_dim=2)

    # matmul as convolution
    W = conv3d.weight.view(conv3d.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


if __name__ == '__main__':
    test_im2col_conv2d()
    test_im2col_conv3d()
