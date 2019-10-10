import torch
from torchsso import DiagCovLinear, KronCovLinear, Fisher


class DiagFisherLinear(DiagCovLinear, Fisher):

    def __init__(self, *args, **kwargs):
        DiagCovLinear.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):
        if self.do_backward:
            assert self.prob is not None

            data_input = getattr(self._module, 'data_input', None)  # n x f_in
            assert data_input is not None

            n = data_input.shape[0]

            in_in = data_input.mul(data_input)  # n x f_in

            pg = torch.mul(grad_output, self.prob.reshape(n, 1))
            grad_grad = pg.mul(grad_output)  # n x f_out

            data_w = torch.einsum('ki,kj->ij', grad_grad,
                                  in_in).div(n)  # f_out x f_in
            self._data = [data_w]

            if self.bias:
                data_b = grad_grad.mean(dim=0)  # f_out x 1
                self._data.append(data_b)

            self.accumulate_cov(self._data)
        else:
            self._data = self.finalize()


class KronFisherLinear(KronCovLinear, Fisher):

    def __init__(self, *args, **kwargs):
        KronCovLinear.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):
        if self.do_backward:
            assert self.prob is not None
            n = grad_output.shape[0]  # n x f_out

            pg = torch.mul(grad_output, self.prob.reshape(n, 1))

            # f_out x f_out
            G = torch.einsum(
                'ki,kj->ij', pg, grad_output).div(n)
            self._G = G
            self.accumulate_cov(G)
        else:
            self._G = self.finalize()

    def update_as_presoftmax(self, prob):
        n, dim = prob.shape
        cov = torch.einsum('ki,kj->ij', prob, prob).div(n)
        fisher_presoftmax = (torch.diag(prob.sum(dim=0)) - cov).div(n)
        self._G = fisher_presoftmax

