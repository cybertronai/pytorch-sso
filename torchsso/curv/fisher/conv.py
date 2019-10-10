from torchsso import DiagCovConv2d, KronCovConv2d, Fisher
import torch
import torch.nn.functional as F


class DiagFisherConv2d(DiagCovConv2d, Fisher):

    def __init__(self, *args, **kwargs):
        DiagCovConv2d.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):

        if self.do_backward:
            assert self.prob is not None

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

            pgi = torch.mul(grad_in, self.prob.reshape(n, 1, 1))
            data_w = pgi.mul(grad_in).mean(dim=0)  # c_out x (c_in)(k_h)(k_w)
            data_w = data_w.reshape((c_out, -1, *conv2d.kernel_size))  # c_out x c_in x k_h x k_w
            self._data = [data_w]

            if self.bias:
                pg = torch.mul(grad_output2d, self.prob.reshape(n, 1, 1))
                grad_grad = pg.mul(grad_output2d)  # n x c_out x (h_out)(w_out)
                data_b = grad_grad.sum(dim=2).mean(dim=0)  # c_out
                self._data.append(data_b)

            self.accumulate_cov(self._data)
        else:
            self._data = self.finalize()


class KronFisherConv2d(KronCovConv2d, Fisher):

    def __init__(self, *args, **kwargs):
        KronCovConv2d.__init__(self, *args, **kwargs)
        Fisher.__init__(self)

    def update_in_backward(self, grad_output):
        if self.do_backward:
            assert self.prob is not None
            n, c, h, w = grad_output.shape  # n x c_out x h_out x w_out

            pg = torch.mul(grad_output, self.prob.reshape(n, 1, 1, 1))
            pm = pg.transpose(0, 1).reshape(c, -1)  # c_out x n(h_out)(w_out)
            m = grad_output.transpose(0, 1).reshape(c, -1)  # c_out x n(h_out)(w_out)

            G = torch.einsum('ik,jk->ij', pm, m).div(n*h*w)  # c_out x c_out
            self._G = G
            self.accumulate_cov(G)
        else:
            self._G = self.finalize()
