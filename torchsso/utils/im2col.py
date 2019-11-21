import torch
import torch.nn as nn
import torch.nn.functional as F


def im2col_2d(x: torch.Tensor, conv2d: nn.Module):
    assert x.ndimension() == 4  # n x c x h_in x w_in
    assert isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d))

    # n x c(k_h)(k_w) x (h_out)(w_out)
    Mx = F.unfold(x, conv2d.kernel_size,
                  dilation=conv2d.dilation,
                  padding=conv2d.padding,
                  stride=conv2d.stride)

    return Mx


def im2col_3d(x: torch.Tensor, conv3d: nn.Conv3d):
    assert x.ndimension() == 5  # n x c x t_in x h_in x w_in
    assert isinstance(conv3d, (nn.Conv3d, nn.ConvTranspose3d))

    kernel_size = conv3d.kernel_size
    stride = conv3d.stride
    assert conv3d.dilation == (1, 1, 1)

    # padding
    pad_left = pad_right = conv3d.padding[0]
    pad_top = pad_bottom = conv3d.padding[1]
    pad_front = pad_back = conv3d.padding[2]
    x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])

    # n x c x t_out x h_out x w_out x k_t x k_h x k_w
    x_slices = x.unfold(
        2, kernel_size[0], stride[0]).unfold(
        3, kernel_size[1], stride[1]).unfold(
        4, kernel_size[2], stride[2])

    # n x c x t_out x h_out x w_out x (k_t)(k_h)(k_w)
    Mx = x_slices.flatten(start_dim=5)
    # n x c x (t_out)(h_out)(w_out) x (k_t)(k_h)(k_w)
    Mx = Mx.flatten(start_dim=2, end_dim=4)
    # n x c x (k_t)(k_h)(k_w) x (t_out)(h_out)(w_out)
    Mx = Mx.transpose(2, 3)
    # n x c(k_t)(k_h)(k_w) x (t_out)(h_out)(w_out)
    Mx = Mx.flatten(start_dim=1, end_dim=2)

    return Mx
