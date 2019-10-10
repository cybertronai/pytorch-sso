import warnings

import numpy
try:
    import cupy
    from torchsso.utils.cupy import to_cupy
except:
    pass
    # print("No cupy detected")

from chainer.backends import cuda

import torch


class Packer(object):

    def __init__(self):
        self.unpack_kernel = cupy.ElementwiseKernel(
            'raw T vec, int32 matrix_size',
            'raw T mat',
            """
            int x = i % matrix_size;
            int y = i / matrix_size;
            if( x < y ) {
                int tmp = y;
                y = x;
                x = tmp;
            }
            mat[i] = vec[matrix_size * y - y * (y + 1) / 2 + x];
            """,
            'unpack'
        )

    def pack(self, arrays, gpu_buf, sizeof_dtype, stream, offset=0):
        buf_offset = offset * sizeof_dtype
        for local_arrays in arrays:
            for array, triangular in local_arrays:
                if triangular:
                    nbytes = self._put_triangular_matrix_to_device_memory(
                        array, gpu_buf, buf_offset, stream)
                else:
                    nbytes = array.size * sizeof_dtype
                    gpu_buf.from_device(array, nbytes, buf_offset, stream)
                buf_offset += nbytes

    def unpack(self, arrays, gpu_buf, sizeof_dtype, stream, offset=0):
        buf_offset = offset * sizeof_dtype
        for local_arrays in arrays:
            for array, triangular in local_arrays:
                if triangular:
                    nbytes = self._get_triangular_matrix_from_device_memory(
                        array, gpu_buf, buf_offset, stream)
                else:
                    nbytes = array.size * sizeof_dtype
                    gpu_buf.to_device(array, nbytes, buf_offset, stream)
                buf_offset += nbytes

    def _put_triangular_matrix_to_device_memory(
            self, array, memory, offset, stream):
        """Puts a triangular matrix to ``DeviceMemory``
        """
        if array.dtype.char == 'f' or array.dtype.char == 'd':
            dtype = array.dtype.char
        else:
            dtype = numpy.find_common_type((array.dtype.char, 'f'), ()).char

        cublas_handle = cupy.cuda.device.get_cublas_handle()

        if array.shape[0] != array.shape[1]:
            raise RuntimeError('non square matrix')

        n = array.shape[0]
        nelems = n * (n + 1) // 2
        nbytes = nelems * array.dtype.itemsize

        if dtype == 'f':
            trttp = cupy.cuda.cublas.strttp
        else:
            trttp = cupy.cuda.cublas.dtrttp

        with stream:
            trttp(cublas_handle, cupy.cuda.cublas.CUBLAS_FILL_MODE_LOWER, n,
                  array.data.ptr, n, memory.ptr() + offset)

        return nbytes

    def _get_triangular_matrix_from_device_memory(
            self, array, memory, offset, stream):
        """Gets a triangular matrix from ``DeviceMemory``
        """
        if array.shape[0] != array.shape[1]:
            raise RuntimeError('non square matrix')

        n = array.shape[0]
        nelems = n * (n + 1) // 2
        nbytes = nelems * array.dtype.itemsize

        with stream:
            self.unpack_kernel(
                memory.array(nelems, offset=offset, dtype=array.dtype),
                n, array, size=n * n)

        return nbytes


def _check_array(array, name):
    xp = cuda.get_array_module(array)
    with cuda.get_device_from_array(array):
        if not array.dtype == xp.float32:
            warnings.warn('non FP32 dtype detected in {}'.format(name))
            array = array.astype(xp.float32)
        if not (array.flags.c_contiguous or array.flags.f_contiguous):
            warnings.warn('non contiguous array detected in {}'.format(name))
            array = xp.ascontiguousarray(array)
    return array


def extract(param_groups, indices, extractors):
    """Extracts arrays from given fisher blocks using indices and extractors

    Args:
        fblocks: List of ``FisherBlock`` instances
        indices: List of ``int``s
        extractors: Callable that extract arrays from a given ``FisherBlock``

    Return:
        List of tuple(array, bool). Second item indicates triangular flag.
    """
    arrays = []
    for local_indices in indices:
        local_arrays = []
        for index in local_indices:
            for extractor in extractors:
                for array in extractor(param_groups[index]):
                    local_arrays.append(array)
        arrays.append(local_arrays)
    return arrays


def extract_attr_from_params(attr, target='params', triangular=False):
    """Extracts arrays from all ``Parameter``s in a given ``FisherBlock``
    """

    def _extract_attr_from_params(group):
        arrays = []
        for param in group[target]:
            x = getattr(param, attr, None)
            if x is not None:
                #x = _check_array(x, fblock.linkname)
                #setattr(param, attr, x)
                x_ten = x.data
                x_cp = to_cupy(x_ten)
                arrays.append((x_cp, triangular))
        return arrays

    return _extract_attr_from_params


def extract_attr_from_curv(attr, triangular=False):
    """Extracts arrays from all ``Parameter``s in a given ``FisherBlock``
    """

    def _extract_attr_from_curv(group):
        arrays = []

        curv = group['curv']
        if curv is None:
            return arrays

        target = getattr(curv, attr, None)
        if target is None:
            if curv.data is not None:
                zeros = []
                for x in curv.data:
                    zeros.append(torch.zeros_like(x))
                setattr(curv, attr, zeros)
                target = getattr(curv, attr)
            else:
                return arrays

        for x in target:
            #x = _check_array(x, fblock.linkname)
            #setattr(param, attr, x)
            x_ten = x.data
            x_cp = to_cupy(x_ten)
            _triangular = triangular and x_cp.ndim == 2 and x_cp.shape[0] == x_cp.shape[1]
            arrays.append((x_cp, _triangular))

        return arrays

    return _extract_attr_from_curv


def get_nelems(arrays):
    """Computes number of elements from given arrays using the triangular flag.
    """
    nelems = 0
    for local_arrays in arrays:
        for array, triangular in local_arrays:
            if triangular:
                if array.shape[0] != array.shape[1]:
                    raise RuntimeError('get_nelems: not a square matrix')
                nelems += array.shape[0] * (array.shape[0] + 1) // 2
            else:
                nelems += array.size
    return nelems


def assign(gpu_buf, nbytes):
    if nbytes > gpu_buf.size:
        gpu_buf.assign(nbytes)
        return True
    return False


def allocate_asgrad(fblocks, attr):
    for fblock in fblocks:
        for _, param in sorted(fblock.link.namedparams()):
            if not hasattr(param, attr):
                # We need to allocate memory space for recieving data
                _grad = param.grad.copy()
                _grad.fill(0.)
                setattr(param, attr, _grad)
