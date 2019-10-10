import torch
from torchsso import KronCurvature


class KronHessian(KronCurvature):

    def update_in_forward(self, input_data):
        raise NotImplementedError

    def update_in_backward(self, grad_output):
        output = getattr(self._module, 'data_output')

        device = grad_output.device
        n = grad_output.shape[0]
        dim = grad_output.shape[1]

        post_curv = self.post_curv

        if post_curv is not None:
            post_module = post_curv.module

        import time

        print('-----------------')
        start = time.time()

        print(self.module)

        if post_curv is not None:
            post_module = post_curv.module
            print(post_module)

            post_output = getattr(post_module, 'data_output')
            post_dim = post_output.shape[1]

            post_out_grad_out = torch.zeros((n, post_dim, dim))  # n x post_dim x dim
            if post_dim <= dim:
                post_output = reshape_4d_to_2d(post_output)
                print('n: {}, dim: {}'.format(len(post_output), post_dim))
                for i in range(post_dim):
                    outputs = tuple(po[i] for po in post_output)
                    grad = torch.autograd.grad(outputs, output, create_graph=True)
                    post_out_grad_out[:, i, :] = reshape_4d_to_2d(grad[0], reduce=True)  # n x dim
            else:
                post_grad_output = getattr(post_module, 'grad_output')
                grad_output = reshape_4d_to_2d(grad_output)
                print('n: {}, dim: {}'.format(len(grad_output), dim))
                for i in range(dim):
                    outputs = tuple(g[i] for g in grad_output)
                    grad = torch.autograd.grad(outputs, post_grad_output, create_graph=True)
                    post_out_grad_out[:, :, i] = reshape_4d_to_2d(grad[0], reduce=True)  # n x post_dim

            post_out_grad_out = post_out_grad_out.to(device)

            recursive_approx = getattr(post_curv, 'recursive_approx', False)
            if recursive_approx:
                equation = 'bij,ik,bkl->bjl'
                post_hessian_output = post_curv.G  # post_dim x post_dim
            else:
                equation = 'bij,bik,bkl->bjl'
                post_hessian_output = getattr(post_module, 'hessian_output', None)  # n x post_dim x post_dim

            msg = 'hessian of loss w.r.t. outputs of post layer' \
                  ' have to be computed beforehand.'
            assert post_hessian_output is not None, msg

            # compute sample hessian_output based on hessian_output of post module
            hessian_output = torch.einsum(equation,
                                          post_out_grad_out,  # n x post_dim x dim
                                          post_hessian_output,  # n x post_dim x post_dim
                                          post_out_grad_out)  # n x post_dim x dim

            del post_module.hessian_output
            del post_out_grad_out

        else:
            # compute sample hessian_output from scratch
            hessian_output = torch.zeros((n, dim, dim))
            print('n: {}, dim: {}'.format(len(grad_output), dim))
            for i in range(dim):
                outputs = tuple(g[i] for g in reshape_4d_to_2d(grad_output))
                grad = torch.autograd.grad(outputs, output, create_graph=True)
                hessian_output[:, i, :] = reshape_4d_to_2d(grad[0], reduce=True)

        hessian_output = hessian_output.to(device)
        setattr(self._module, 'hessian_output', hessian_output)

        # refresh hessian_output
        self._G = hessian_output.sum((0,))  # dim x dim

        elapsed = time.time() - start
        print('{}s'.format(elapsed))

    def precondition_grad(self, params):
        raise NotImplementedError

    def sample_params(self, params, mean, std_scale):
        raise NotImplementedError

    def backward_postprocess(self, module, grad_input, grad_output):
        # skip hook for higher order derivative
        order = getattr(module, 'derivative_order', 1)
        if order > 1:
            return

        super(KronHessian, self).backward_postprocess(module, grad_input, grad_output)

        # skip hook for higher order derivative
        setattr(module, 'derivative_order', 2)

    def reset_derivative_order(self):
        module = self._module
        setattr(module, 'derivative_order', 1)

    def step(self, update_std=False):
        super(KronHessian, self).step(update_std)
        self.reset_derivative_order()


def reshape_4d_to_2d(data, reduce=False):
    ndim = len(data.shape)
    if ndim == 2:
        return data

    assert ndim == 4, 'number of dimension of data is expected to be 4, got {}.'.format(ndim)

    if reduce:
        # n x c x h x w -> n x c
        return data.sum((2, 3))
    else:
        n, c, h, w = data.shape
        # n x c x h x w -> n x h x w x c -> n*h*w x c
        data = data.transpose(1, 2).transpose(2, 3).contiguous().view(n*h*w, c)
        return data

