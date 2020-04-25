import torch
from torchsso.autograd.utils import original_requires_grad, matrix_to_tril, extend_A_tril

OP_KRON_COV = 'kron_cov'
OP_DIAG_COV = 'diag_cov'
OP_BATCH_GRADS = 'batch_grads'


class Operation:

    def __init__(self, module, op_names, save_attr='op_results'):
        self._module = module
        self._op_names = op_names
        self._save_attr = save_attr
        self._grads_scale = None

    def get_op_results(self):
        return getattr(self._module, self._save_attr, {})

    def delete_op_results(self):
        if hasattr(self._module, self._save_attr):
            delattr(self._module, self._save_attr)

    @property
    def grads_scale(self):
        return self._grads_scale

    @grads_scale.setter
    def grads_scale(self, value):
        self._grads_scale = value

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_KRON_COV in self._op_names:
            A = self.kron_cov_A(module, in_data)
            A_tril = matrix_to_tril(A)  # save only lower triangular
            if original_requires_grad(module, 'bias'):
                A_tril = extend_A_tril(A_tril)
            op_results = self.get_op_results()
            op_results[OP_KRON_COV] = {'A_tril': A_tril}
            setattr(module, self._save_attr, op_results)

    def backward_pre_process(self, in_data, out_grads):
        module = self._module
        op_results = self.get_op_results()

        if self._grads_scale is not None:
            shape = (-1,) + (1,) * (out_grads.ndim - 1)
            out_grads = torch.mul(out_grads, self._grads_scale.reshape(shape))

        for op_name in self._op_names:
            if op_name == OP_KRON_COV:
                B = self.kron_cov_B(module, out_grads)
                B_tril = matrix_to_tril(B)  # save only lower triangular
                op_results[OP_KRON_COV] = {'B_tril': B_tril}
            else:
                rst = getattr(self, f'{op_name}_weight')(module, in_data, out_grads)
                op_results[op_name] = {'weight': rst}
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    op_results[op_name]['bias'] = rst

        setattr(module, self._save_attr, op_results)

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

