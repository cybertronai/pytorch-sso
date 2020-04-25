import torch
from torchsso.autograd.utils import requires_grad, matrix_to_tril, extend_A_tril

OP_KRON_COV = 'kron_cov'
OP_DIAG_COV = 'diag_cov'
OP_BATCH_GRADS = 'batch_grads'


class Operation:

    def __init__(self, module, op_names):
        self._module = module
        self._op_names = op_names

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_KRON_COV in self._op_names:
            A = self.kron_cov_A(module, in_data)
            A_tril = matrix_to_tril(A)
            if requires_grad(module, 'bias'):
                A_tril = extend_A_tril(A_tril)
            setattr(module, OP_KRON_COV, {'A_tril': A_tril})

    def backward_pre_process(self, in_data, out_grads):
        module = self._module
        for op_name in self._op_names:
            if op_name == OP_KRON_COV:
                B = self.kron_cov_B(module, out_grads)
                B_tril = matrix_to_tril(B)
                getattr(module, OP_KRON_COV)['B_tril'] = B_tril
            else:
                rst = getattr(self, f'{op_name}_weight')(module, in_data, out_grads)
                setattr(module.weight, op_name, rst)
                if requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    setattr(module.bias, op_name, rst)

    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def diag_cov_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def diag_cov_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def kron_cov_A(module, in_data):
        raise NotImplementedError

    @staticmethod
    def kron_cov_B(module, out_grads):
        raise NotImplementedError

