import torch
import torch.nn as nn

from torchsso.utils import *


def test_im2col_convnd(ndim=2, n=10, c_in=3, c_out=5, k=3, s=2, p=1, d=28, thr=1e-6):

    if ndim == 1:
        x = torch.randn(n, c_in, d)
        conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
    elif ndim == 2:
        x = torch.randn(n, c_in, d, d)
        conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
    else:
        x = torch.randn(n, c_in, d, d, d)
        conv = nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)

    output = conv(x)

    # im2col
    Mx = im2col(x, conv)

    # matmul as convolution
    W = conv.weight.view(conv.weight.size(0), -1)
    Mout = W.matmul(Mx)
    output_test = Mout.view(output.size())

    error = output - output_test
    ratio = error.norm() / output.norm()
    assert ratio < thr


if __name__ == '__main__':
    c_in, c_out = 3, 5
    for k in range(1, 5):
        for s in range(1, 5):
            for p in range(1, k):
                kwargs = dict(k=k, s=s, p=p)
                print(kwargs)
                test_im2col_convnd(ndim=1, **kwargs)
                test_im2col_convnd(ndim=2, **kwargs)
                test_im2col_convnd(ndim=3, **kwargs)
