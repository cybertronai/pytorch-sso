import torch
from torch import nn

from torchsso.autograd.operations import Operation


class Linear(Operation):

    @staticmethod
    def batch_grads_weight(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        return torch.bmm(out_grads.unsqueeze(2), in_data.unsqueeze(1))  # n x f_out x f_in

    @staticmethod
    def diag_cov_weight(module, in_data, out_grads):
        n = in_data.shape[0]
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in).div(n)

    @staticmethod
    def diag_cov_bias(module, out_grads):
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return grad_grad.mean(dim=0)  # f_out x 1

    @staticmethod
    def kron_cov_A(module, in_data):
        n = in_data.shape[0]
        return torch.matmul(in_data.T, in_data).div(n)  # f_in x f_in

    @staticmethod
    def kron_cov_B(module, out_grads):
        n = out_grads.shape[0]
        return torch.matmul(out_grads.T, out_grads).div(n)  # f_in x f_in
