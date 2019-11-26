from torchsso.curv import DiagCurvature, KronCurvature
from torchsso.utils import im2col, grad2col
import torch
import torch.nn as nn


class _ConvMixin(object):

    @staticmethod
    def get_input2d(data_input: torch.Tensor, module: nn.Module):
        # kernel_size = (k_1)(k_2)...(k_d)
        input2d = im2col(data_input, module)

        return input2d  # n x (c_in)(kernel_size) x output_size

    @staticmethod
    def get_grad_output2d(grad_output: torch.Tensor, module: nn.Module):
        n, c_out = grad_output.size()[0:2]
        grad_output2d = grad_output.view(n, c_out, -1)

        return grad_output2d  # n x c_out x output_size


class DiagCovConvNd(_ConvMixin, DiagCurvature):

    def update_in_backward(self, data_input: torch.Tensor, grad_output: torch.Tensor):
        module = self._module

        if self.weight_requires_grad:
            input2d = self.get_input2d(data_input, module)
            grad_output2d = self.get_grad_output2d(grad_output, module)
            grads_2d = self.get_weight_grad2d(input2d, grad_output2d)  # n x c_out x (c_in)(kernel_size)
            data_w = grads_2d.mul(grads_2d).mean(dim=0)  # c_out x (c_in)(kernel_size)
            data_w = data_w.reshape(module.weight.size())  # c_out x c_in x k_1 x ... x k_d
            self._data = [data_w]

        if self.bias_requires_grad:
            ndim = grad_output.ndimension()
            grad_grad = grad_output.mul(grad_output)  # n x c_out x output_size
            data_b = grad_grad.sum(dim=tuple(range(ndim))[2:]).mean(dim=0)  # c_out
            self._data.append(data_b)

    @staticmethod
    def get_weight_grad2d(input2d: torch.Tensor, grad_output2d: torch.Tensor):
        grad_2d = torch.einsum('bik,bjk->bij', grad_output2d, input2d)
        return grad_2d  # n x c_out x (c_in)(kernel_size)


class DiagCovConvTransposeNd(DiagCovConvNd):

    @staticmethod
    def get_input2d(data_input: torch.Tensor, module: nn.Module):
        n, c_in = data_input.size()[0:2]
        input2d = data_input.view(n, c_in, -1)

        return input2d  # n x c_in x input_size

    @staticmethod
    def get_grad_output2d(grad_output: torch.Tensor, module: nn.Module):
        # kernel_size = (k_1)(k_2)...(k_d)
        grad_output2d = im2col(grad_output, module)

        return grad_output2d  # n x (c_out)(kernel_size) x input_size

    @staticmethod
    def get_weight_grad2d(input2d: torch.Tensor, grad_output2d: torch.Tensor):
        grad_2d = torch.einsum('bik,bjk->bij', input2d, grad_output2d)
        return grad_2d  # n x c_in x (c_out)(kernel_size)


class KronCovConvNd(_ConvMixin, KronCurvature):

    def update_in_forward(self, data_input):
        module = self._module
        input2d = self.get_input2d(data_input, module)
        n, a, _ = input2d.shape  # n x (c_in)(kernel_size) x output_size
        m = input2d.transpose(0, 1).reshape(a, -1)  # (c_in)(kernel_size) x n(output_size)

        if self.bias_requires_grad:
            b = m.size()[-1]
            m = torch.cat((m, m.new_ones((1, b))), 0)

        A = torch.einsum('ik,jk->ij', m, m).div(n)  # (c_in)(kernel_size) x (c_in)(kernel_size)
        self._A = A

    def update_in_backward(self, data_input, grad_output):
        module = self._module
        grad_output2d = self.get_grad_output2d(grad_output, module)
        m = grad_output2d.transpose(0, 1).flatten(start_dim=1)  # c_out x n(output_size)

        G = torch.einsum('ik,jk->ij', m, m).div(m.size(-1))  # c_out x c_out
        self._G = G

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv
        c_out = params[0].size(0)

        if self.bias_requires_grad:
            weight_grad2d = params[0].grad.reshape(c_out, -1)  # c_out x (c_in)(kernel_size)
            bias_grad1d = params[1].grad.view(c_out, 1)  # c_out x 1
            grad2d = torch.cat((weight_grad2d, bias_grad1d), 1)  # c_out x {(c_in)(kernel_size) + 1}
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad2d[:, 0:-1].reshape_as(params[0]))
            params[1].grad.copy_(preconditioned_grad2d[:, -1])
        else:
            grad2d = params[0].grad.reshape(c_out, -1)
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad2d.reshape_as(params[0]))

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std
        c_out = params[0].size(0)

        if self.bias_requires_grad:
            m = torch.cat(
                (mean[0].reshape(c_out, -1), mean[1].view(-1, 1)), 1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data.copy_(param[:, 0:-1].reshape(params[0].size()))
            params[1].data.copy_(param[:, -1])
        else:
            m = mean[0].reshape(c_out, -1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data = param.reshape(params[0].size())

    def _get_shape(self):
        weight = self._module.weight
        c_out, c_in, kernel_size = weight.flatten(start_dim=2).size()

        G_shape = (c_out, c_out)

        dim = c_in * kernel_size
        if self.bias_requires_grad:
            A_shape = (dim + 1, dim + 1)
        else:
            A_shape = (dim, dim)

        return A_shape, G_shape


class KronCovConvTransposeNd(KronCovConvNd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_input2d(data_input: torch.Tensor, module: nn.Module):
        input2d = grad2col(data_input, module)

        return input2d  # n x (c_in)(kernel_size) x output_size

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv
        c_in, c_out = params[0].size()[0:2]

        if self.bias_requires_grad:
            weight_grad2d = params[0].grad.transpose(0, 1).reshape(c_out, -1)  # c_out x (c_in)(kernel_size)
            bias_grad1d = params[1].grad.view(c_out, 1)  # c_out x 1
            grad2d = torch.cat((weight_grad2d, bias_grad1d), 1)  # c_out x {(c_in)(kernel_size) + 1}
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)

            pgW = preconditioned_grad2d[:, :-1].view(c_out, c_in, -1).transpose(0, 1)  # c_in x c_out x kernel_size
            params[0].grad.copy_(pgW.reshape_as(params[0]))  # c_in x c_out x k_1 x ... x k_d
            params[1].grad.copy_(preconditioned_grad2d[:, -1])
        else:
            grad2d = params[0].grad.reshape(c_out, -1)
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)  # c_out x (c_in)(kernel_size)
            pg = preconditioned_grad2d.view(c_out, c_in, -1).transpose(0, 1)  # c_in x c_out x kernel_size
            params[0].grad.copy_(pg.reshape_as(params[0]))

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std
        c_in, c_out = params[0].size()[0:2]

        if self.bias_requires_grad:
            m_W = mean[0].transpose(0, 1).view(c_out, -1)  # c_out x (c_in)(kernel_size)
            m_b = mean[1].view(c_out, 1)  # c_out x 1
            m = torch.cat((m_W, m_b), 1)  # c_out x {(c_in)(kernel_size) + 1}
            noise = G_ic.mm(torch.randn_like(m)).mm(A_ic)  # c_out x {(c_in)(kernel_size) + 1}

            noise_W = noise[:, :-1].view(c_out, c_in, -1).transpose(0, 1)  # c_in x c_out x kernel_size
            params[0].data.copy_(mean[0].add(std_scale, noise_W.reshape_as(mean[0])))
            params[1].data.copy_(mean[1].add(std_scale, noise[:, :-1]))
        else:
            m = mean[0].transpose(0, 1).view(c_out, -1)  # c_out x (c_in)(kernel_size)
            noise = G_ic.mm(torch.randn_like(m)).mm(A_ic)  # c_out x (c_in)(kernel_size)
            noise = noise.view(c_out, c_in, -1).transpose(0, 1)  # c_in x c_out x kernel_size
            params[0].data.copy_(mean[0].add(std_scale, noise.reshape_as(mean[0])))  # c_in x c_out x k_1 x ... x k_d

    def _get_shape(self):
        weight = self._module.weight
        c_in, c_out, kernel_size = weight.flatten(start_dim=2).size()

        G_shape = (c_out, c_out)

        dim = c_in * kernel_size
        if self.bias_requires_grad:
            A_shape = (dim + 1, dim + 1)
        else:
            A_shape = (dim, dim)

        return A_shape, G_shape
