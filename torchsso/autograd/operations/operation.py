import torch
from torchsso.autograd.utils import original_requires_grad

OP_KRON = 'kron'
OP_DIAG = 'diag'
OP_BATCH_GRADS = 'batch_grads'


class Operation:

    def __init__(self, module, op_names, save_attr='op_results'):
        self._module = module
        self._op_names = op_names
        self._save_attr = save_attr
        self._grads_scale = None
        self._accumulate_out_grads = False
        self._out_grads_list = []

    def get_op_results(self):
        return getattr(self._module, self._save_attr, {})

    def set_op_results(self, op_results):
        setattr(self._module, self._save_attr, op_results)

    def delete_op_results(self):
        if hasattr(self._module, self._save_attr):
            delattr(self._module, self._save_attr)

    @property
    def grads_scale(self):
        return self._grads_scale

    @grads_scale.setter
    def grads_scale(self, value):
        self._grads_scale = value

    def turn_on_out_grads_accumulation(self):
        self._accumulate_out_grads = True

    def turn_off_out_grads_accumulation(self):
        self._accumulate_out_grads = False

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_KRON in self._op_names:
            A = self.kron_A(module, in_data)
            op_results = self.get_op_results()
            op_results[OP_KRON] = {'A': A}
            self.set_op_results(op_results)

    def backward_pre_process(self, in_data, out_grads):
        if self._grads_scale is not None:
            shape = (-1,) + (1,) * (out_grads.ndim - 1)
            out_grads = torch.mul(out_grads, self._grads_scale.reshape(shape))

        multi_outputs = False
        if self._accumulate_out_grads:
            self._out_grads_list.append(out_grads)
            return
        elif len(self._out_grads_list) > 0:
            self._out_grads_list.append(out_grads)
            out_grads = torch.cat(self._out_grads_list, dim=0)
            multi_outputs = True
            self._out_grads_list = []

        module = self._module
        op_results = self.get_op_results()
        for op_name in self._op_names:
            if op_name == OP_KRON:
                B = self.kron_B(module, out_grads)
                if OP_KRON in op_results:
                    op_results[OP_KRON]['B'] = B
                else:
                    op_results[OP_KRON] = {'B': B}
            else:
                if multi_outputs:
                    func_name = f'{op_name}_weight_multi_outputs'
                else:
                    func_name = f'{op_name}_weight'
                rst = getattr(self, func_name)(module, in_data, out_grads)
                op_results[op_name] = {'weight': rst}
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    op_results[op_name]['bias'] = rst

        self.set_op_results(op_results)

    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_weight_multi_outputs(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_bias(module, out_grads):
        return out_grads

    @staticmethod
    def diag_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def diag_weight_multi_outputs(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def diag_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def kron_A(module, in_data):
        raise NotImplementedError

    @staticmethod
    def kron_B(module, out_grads):
        raise NotImplementedError

