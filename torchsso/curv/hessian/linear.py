from torchsso import KronCovLinear, KronHessian


class KronHessianLinear(KronCovLinear, KronHessian):

    def __init__(self, module, ema_decay=1., damping=0, post_curv=None, recursive_approx=False):
        KronHessian.__init__(self, module, ema_decay, damping, post_curv, recursive_approx)

    def update_in_backward(self, grad_output):
        KronHessian.update_in_backward(self, grad_output)
