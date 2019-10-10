import torch
from torchsso import DiagCovBatchNorm2d, Fisher


class DiagFisherBatchNorm2d(DiagCovBatchNorm2d, Fisher):

    def __init__(self, *args, **kwargs):
        DiagCovBatchNorm2d.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_out):
        if self.do_backward:
            assert self.prob is not None
            data_input = getattr(self._module, 'data_input', None)  # n x c x h x w
            assert data_input is not None

            n = grad_out.shape[0]  # n x c x h x w
            pg = torch.mul(grad_out, self.prob.reshape(n, 1, 1, 1))

            grad_grad = pg.mul(grad_out).sum(dim=(2, 3))  # n x c
            in_in = data_input.mul(data_input).sum(dim=(2, 3))  # n x c

            data_w = in_in.mul(grad_grad).mean(dim=0)  # c x 1

            self._data = [data_w]

            if self.bias:
                data_b = grad_grad.mean(dim=0)  # c x 1
                self._data.append(data_b)
            self.accumulate_cov(self._data)
        else:
            self._data = self.finalize()
