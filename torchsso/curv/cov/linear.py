import torch
from torchsso import Curvature, DiagCurvature, KronCurvature


class CovLinear(Curvature):

    def update_in_backward(self, grad_output):
        data_input = getattr(self._module, 'data_input', None)  # n x f_in
        assert data_input is not None

        n = data_input.shape[0]

        if self.bias:
            ones = torch.ones((n, 1), device=data_input.device, dtype=data_input.dtype)
            data_input = torch.cat((data_input, ones), 1)  # n x (f_in+1)

        grad = torch.einsum('bi,bj->bij', grad_output, data_input)  # n x f_out x f_in
        grad = grad.reshape((n, -1))  # n x (f_out)(f_in)

        data = torch.einsum('bi,bj->ij', grad, grad)

        self._data = [data]

    def precondition_grad(self, params):
        pass


class DiagCovLinear(DiagCurvature):

    def update_in_backward(self, grad_output):
        data_input = getattr(self._module, 'data_input', None)  # n x f_in
        assert data_input is not None

        n = data_input.shape[0]

        in_in = data_input.mul(data_input)  # n x f_in
        grad_grad = grad_output.mul(grad_output)  # n x f_out

        data_w = torch.einsum('ki,kj->ij', grad_grad,
                              in_in).div(n)  # f_out x f_in
        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # f_out x 1
            self._data.append(data_b)


class KronCovLinear(KronCurvature):

    def update_in_forward(self, input_data):
        n = input_data.shape[0]  # n x f_in
        if self.bias:
            ones = input_data.new_ones((n, 1))
            # shape: n x (f_in+1)
            input_data = torch.cat((input_data, ones), 1)

        # f_in x f_in or (f_in+1) x (f_in+1)
        A = torch.einsum('ki,kj->ij', input_data, input_data).div(n)
        self._A = A

    def update_in_backward(self, grad_output):
        n = grad_output.shape[0]  # n x f_out

        # f_out x f_out
        G = torch.einsum(
            'ki,kj->ij', grad_output, grad_output).div(n)
        self._G = G

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        if self.bias:
            grad = torch.cat(
                (params[0].grad, params[1].grad.view(-1, 1)), 1)
            preconditioned_grad = G_inv.mm(grad).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad[:, :-1])
            params[1].grad.copy_(preconditioned_grad[:, -1])
        else:
            grad = params[0].grad
            preconditioned_grad = G_inv.mm(grad).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad)

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std

        if self.bias:
            m = torch.cat(
                (mean[0], mean[1].view(-1, 1)), 1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data.copy_(param[:, 0:-1])
            params[1].data.copy_(param[:, -1])
        else:
            m = mean[0]
            param = mean.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data = param

    def _get_shape(self):
        linear = self._module
        w = getattr(linear, 'weight')
        f_out, f_in = w.shape

        G_shape = (f_out, f_out)

        if self.bias:
            A_shape = (f_in + 1, f_in + 1)
        else:
            A_shape = (f_in, f_in)

        return A_shape, G_shape

