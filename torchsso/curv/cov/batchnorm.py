from torchsso import Curvature, DiagCurvature


class CovBatchNorm1d(Curvature):

    def update_in_backward(self, grad_output_data):
        pass


class DiagCovBatchNorm1d(DiagCurvature):

    def update_in_backward(self, grad_output):
        data_input = getattr(self._module, 'data_input', None)  # n x f
        assert data_input is not None

        in_in = data_input.mul(data_input)  # n x f
        grad_grad = grad_output.mul(grad_output)  # n x f

        data_w = in_in.mul(grad_grad).mean(dim=0)  # f x 1

        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # f x 1
            self._data.append(data_b)


class CovBatchNorm2d(Curvature):

    def update_in_backward(self, grad_output):
        pass


class DiagCovBatchNorm2d(DiagCurvature):

    def update_in_backward(self, grad_out):
        data_input = getattr(self._module, 'data_input', None)  # n x c x h x w
        assert data_input is not None

        in_in = data_input.mul(data_input).sum(dim=(2, 3))  # n x c
        grad_grad = grad_out.mul(grad_out).sum(dim=(2, 3))  # n x c

        data_w = in_in.mul(grad_grad).mean(dim=0)  # c x 1

        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # c x 1
            self._data.append(data_b)
