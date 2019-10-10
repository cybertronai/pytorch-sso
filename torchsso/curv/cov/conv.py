from torchsso import Curvature, DiagCurvature, KronCurvature
import torch
import torch.nn.functional as F


class CovConv2d(Curvature):

    def update_in_backward(self, grad_output):
        pass

    def precgrad(self, params):
        pass


class DiagCovConv2d(DiagCurvature):

    def update_in_backward(self, grad_output):
        conv2d = self._module
        data_input = getattr(conv2d, 'data_input', None)  # n x c_in x h_in x w_in
        assert data_input is not None

        # n x (c_in)(k_h)(k_w) x (h_out)(w_out)
        input2d = F.unfold(data_input,
                           kernel_size=conv2d.kernel_size, stride=conv2d.stride,
                           padding=conv2d.padding, dilation=conv2d.dilation)

        # n x c_out x h_out x w_out
        n, c_out, h, w = grad_output.shape
        # n x c_out x (h_out)(w_out)
        grad_output2d = grad_output.reshape(n, c_out, -1)

        grad_in = torch.einsum('bik,bjk->bij',
                               grad_output2d, input2d)  # n x c_out x (c_in)(k_h)(k_w)

        data_w = grad_in.mul(grad_in).mean(dim=0)  # c_out x (c_in)(k_h)(k_w)
        data_w = data_w.reshape((c_out, -1, *conv2d.kernel_size))  # c_out x c_in x k_h x k_w
        self._data = [data_w]

        if self.bias:
            grad_grad = grad_output2d.mul(grad_output2d)  # n x c_out x (h_out)(w_out)
            data_b = grad_grad.sum(dim=2).mean(dim=0)  # c_out
            self._data.append(data_b)


class KronCovConv2d(KronCurvature):

    def update_in_forward(self, data_input):
        conv2d = self._module

        # n x (c_in)(k_h)(k_w) x (h_out)(w_out)
        input2d = F.unfold(data_input,
                           kernel_size=conv2d.kernel_size, stride=conv2d.stride,
                           padding=conv2d.padding, dilation=conv2d.dilation)

        n, a, _ = input2d.shape

        # (c_in)(k_h)(k_w) x n(h_out)(w_out)
        m = input2d.transpose(0, 1).reshape(a, -1)
        a, b = m.shape
        if self.bias:
            # {(c_in)(k_h)(k_w) + 1} x n(h_out)(w_out)
            m = torch.cat((m, m.new_ones((1, b))), 0)

        # (c_in)(k_h)(k_w) x (c_in)(k_h)(k_w) or
        # {(c_in)(k_h)(k_w) + 1} x {(c_in)(k_h)(k_w) + 1}
        A = torch.einsum('ik,jk->ij', m, m).div(n)
        self._A = A

    def update_in_backward(self, grad_output):
        n, c, h, w = grad_output.shape  # n x c_out x h_out x w_out
        m = grad_output.transpose(0, 1).reshape(c, -1)  # c_out x n(h_out)(w_out)

        G = torch.einsum('ik,jk->ij', m, m).div(n*h*w)  # c_out x c_out
        self._G = G

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        oc, _, _, _ = params[0].shape
        if self.bias:
            grad2d = torch.cat(
                (params[0].grad.reshape(oc, -1), params[1].grad.view(-1, 1)), 1)
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad2d[:, 0:-1].reshape_as(params[0]))
            params[1].grad.copy_(preconditioned_grad2d[:, -1])
        else:
            grad2d = params[0].grad.reshape(oc, -1)
            preconditioned_grad2d = G_inv.mm(grad2d).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad2d.reshape_as(params[0]))

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std
        oc, ic, h, w = mean[0].shape
        if self.bias:
            m = torch.cat(
                (mean[0].reshape(oc, -1), mean[1].view(-1, 1)), 1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data.copy_(param[:, 0:-1].reshape(oc, ic, h, w))
            params[1].data.copy_(param[:, -1])
        else:
            m = mean[0].reshape(oc, -1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data = param.reshape(oc, ic, h, w)

    def _get_shape(self):
        linear = self._module
        w = getattr(linear, 'weight')
        c_out, c_in, k_h, k_w = w.shape

        G_shape = (c_out, c_out)

        dim = c_in * k_h * k_w
        if self.bias:
            A_shape = (dim + 1, dim + 1)
        else:
            A_shape = (dim, dim)

        return A_shape, G_shape

