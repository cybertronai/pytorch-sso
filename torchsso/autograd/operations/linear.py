import torch
from torch import nn

from torchsso.autograd.operations import Operation


class Linear(Operation):

    @staticmethod
    def batch_grads_weight(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        return torch.bmm(out_grads.unsqueeze(2), in_data.unsqueeze(1))  # n x f_out x f_in

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def diag_weight(module, in_data, out_grads):
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return torch.matmul(grad_grad.T, in_in)  # f_out x f_in

    @staticmethod
    def diag_weight_multi_outputs(module, in_data, out_grads):
        n = in_data.shape[0]
        f_out = out_grads.shape[-1]
        in_in = in_data.mul(in_data)  # n x f_in
        grad_grad = out_grads.mul(out_grads)  # cn x f_out
        grad_grad = grad_grad.view(-1, n, f_out)  # c x n x f_out
        rst = torch.matmul(grad_grad.transpose(1, 2), in_in)  # c x f_out x f_in
        return rst.sum(dim=0)  # f_out x f_in

    @staticmethod
    def diag_bias(module, out_grads):
        grad_grad = out_grads.mul(out_grads)  # n x f_out
        return grad_grad.sum(dim=0)  # f_out x 1

    @staticmethod
    def kron_A(module, in_data):
        return torch.matmul(in_data.T, in_data)  # f_in x f_in

    @staticmethod
    def kron_B(module, out_grads):
        return torch.matmul(out_grads.T, out_grads)  # f_out x f_out
