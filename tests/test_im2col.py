import torch
import torch.nn as nn
from  torchsso.utils import im2col_1d, im2col_2d, im2col_3d


def test_im2col_conv1d(n=10, thr=1e-6):
    c_in, c_out, k = 3, 5, 3
    x = torch.randn(n, c_in, 28)
    conv1d = nn.Conv1d(c_in, c_out, k, bias=False, padding=3)
    output = conv1d(x)

    # im2col
    Mx = im2col_1d(x, conv1d)

    # matmul as convolution
    W = conv1d.weight.view(conv1d.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


def test_im2col_conv2d(n=10, thr=1e-6):
    c_in, c_out, k = 3, 5, 3
    x = torch.randn(n, c_in, 28, 28)
    conv2d = nn.Conv2d(c_in, c_out, k, bias=False)
    output = conv2d(x)

    # im2col
    Mx = im2col_2d(x, conv2d)

    # matmul as convolution
    W = conv2d.weight.view(conv2d.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


def test_im2col_conv3d(n=10, thr=1e-6):
    c_in, c_out, k = 4, 5, 3
    x = torch.randn(n, c_in, 28, 28, 28)
    conv3d = nn.Conv3d(c_in, c_out, 3, bias=False, padding=3)
    output = conv3d(x)

    # im2col
    Mx = im2col_3d(x, conv3d)

    # matmul as convolution
    W = conv3d.weight.view(conv3d.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


if __name__ == '__main__':
    test_im2col_conv1d()
    test_im2col_conv2d()
    test_im2col_conv3d()
