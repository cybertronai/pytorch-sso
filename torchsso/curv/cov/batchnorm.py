from torchsso.curv import DiagCurvature
import torch


class DiagCovBatchNormNd(DiagCurvature):

    def update_in_backward(self, data_input: torch.Tensor, grad_output: torch.Tensor):

        ndim = grad_output.ndimension()
        if ndim > 2:
            grad_grad = grad_output.mul(grad_output).sum(dim=tuple(range(ndim))[2:])  # n x c
        else:
            grad_grad = grad_output.mul(grad_output)  # n x f

        if self.weight_requires_grad:
            if ndim > 2:
                in_in = data_input.mul(data_input).sum(dim=tuple(range(ndim))[2:])  # n x c
            else:
                in_in = data_input.mul(data_input)  # n x f

            data_w = in_in.mul(grad_grad).mean(dim=0)  # c x 1

            self._data = [data_w]

        if self.bias_requires_grad:
            data_b = grad_grad.mean(dim=0)  # c x 1
            self._data.append(data_b)
