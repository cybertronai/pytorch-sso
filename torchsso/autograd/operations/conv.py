import torch
from torch import nn

from torchsso.utils import im2col_2d
from torchsso.autograd.operations import Operation


def get_in_data_2d(in_data, module):
    return im2col_2d(in_data, module)


def get_out_grads_2d(out_grads: torch.Tensor):
    n, c_out = out_grads.size()[0:2]
    out_grads_2d = out_grads.view(n, c_out, -1)  # n x c_out x output_size
    return out_grads_2d


class Conv2d(Operation):

    @staticmethod
    def batch_grads_weight(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        # kernel_size = (k_1)(k_2)...(k_d)
        # n x (c_in)(kernel_size) x output_size
        in_data_2d = get_in_data_2d(in_data, module)
        n, c_out = out_grads.size()[0:2]
        out_grads_2d = out_grads.view(n, c_out, -1)  # n x c_out x output_size
        grads_2d = torch.bmm(out_grads_2d, in_data_2d.transpose(2, 1))  # n x c_out x (c_in)(kernel_size)
        return grads_2d.view(n, *module.weight.size())  # n x c_out x c_in x k_1 x ... x k_d

    @staticmethod
    def batch_grads_bias(module: nn.Module, out_grads: torch.Tensor):
        ndim = out_grads.ndim
        return torch.sum(out_grads, dim=tuple(range(ndim))[2:])  # n x c_out

    @staticmethod
    def diag_weight(module, in_data, out_grads):
        in_data_2d = get_in_data_2d(in_data, module)
        out_grads_2d = get_out_grads_2d(out_grads)
        grads_2d = torch.bmm(out_grads_2d, in_data_2d.transpose(2, 1))  # n x c_out x (c_in)(kernel_size)
        rst = grads_2d.mul(grads_2d).sum(dim=0)  # c_out x (c_in)(kernel_size)
        return rst.view_as(module.weight)  # c_out x c_in x k_1 x ... x k_d

    @staticmethod
    def diag_weight_multi_outputs(module, in_data, out_grads):
        n = in_data.shape[0]
        in_data_2d = get_in_data_2d(in_data, module)
        cn, c_out = out_grads.shape[0:2]
        c = int(cn / n)
        out_grads_2d = out_grads.view(c, n, c_out, -1)  # c x n x c_out x output_size

        grads_2d = torch.matmul(out_grads_2d, in_data_2d.transpose(2, 1))  # c x n x c_out x (c_in)(kernel_size)
        rst = grads_2d.mul(grads_2d).sum(dim=(0, 1))  # c_out x (c_in)(kernel_size)
        return rst.view_as(module.weight)  # c_out x c_in x k_1 x ... x k_d

    @staticmethod
    def diag_bias(module, out_grads):
        ndim = out_grads.ndim
        grad_grad = out_grads.mul(out_grads)  # n x c_out x output_size
        return grad_grad.sum(dim=tuple(range(ndim))[2:]).mean(dim=0)  # c_out

    @staticmethod
    def kron_A(module, in_data):
        in_data_2d = get_in_data_2d(in_data, module)
        n, a, b = in_data_2d.shape  # n x (c_in)(kernel_size) x output_size
        m = in_data_2d.transpose(0, 1).reshape(a, -1)  # (c_in)(kernel_size) x n(output_size)
        return torch.matmul(m, m.T).div(b)  # (c_in)(kernel_size) x (c_in)(kernel_size)

    @staticmethod
    def kron_B(module, out_grads):
        out_grads_2d = get_out_grads_2d(out_grads)  # n x c_out x output_size
        m = out_grads_2d.transpose(0, 1).flatten(start_dim=1)  # c_out x n(output_size)
        return torch.matmul(m, m.T)  # c_out x c_out

