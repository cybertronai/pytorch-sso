import itertools
import math

from mpi4py import MPI

import numpy as np
import cupy

import chainer

from chainermn.communicators import _memory_utility
from chainermn.communicators import _communication_utility
from chainermn import nccl

try:
    from hiercoll.hiernccl import HierNcclCommunicator

    _hiercoll_available = True
except ImportError as e:
    _hiercoll_available = False
    _hiercoll_available_exception = e

from torchsso.utils.chainer_communicators import base
from torchsso.utils.chainer_communicators import _utility

NUM_STREAMS = 12


class PureNCCLCommunicator(base.KFACCommunicatorBase):

    def __init__(self,
                 mpi_comm,
                 rsv_comm_dtype=np.float32,
                 agv_comm_dtype=np.float32,
                 use_hiercoll=False,
                 dims=None
                 ):
        super(PureNCCLCommunicator, self).__init__(mpi_comm)

        if use_hiercoll:
            if not _hiercoll_available:
                raise ValueError('use_hiercoll is True,'
                                 'but hiercoll.hiernccl is not available.')

            if dims is None:
                dims = []

        if dims is not None and not use_hiercoll:
            raise ValueError('dim is not None,'
                             'but use_hiercoll is False.')

        if use_hiercoll and mpi_comm.size != MPI.COMM_WORLD.size:
            raise ValueError(
                'HierColl with non-WORLD MPI Comm is not supported.')

        # None -> Non-hierarchical / pure NCCL
        # []   -> auto hierarchical selection (envv or optimizer)
        # [int]-> manual hierarchical selection
        self.dims = dims

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.nccl_comm = None

        # GPU buffers
        self.gpu_buf_a = _memory_utility.DeviceMemory()
        self.gpu_buf_b = _memory_utility.DeviceMemory()
        self.gpu_buf_c = _memory_utility.DeviceMemory()

        # Assume FP32 for data type
        self._arrs_dtype = np.dtype(np.float32)

        # Data type used in communications
        self._rsv_comm_dtype = np.dtype(rsv_comm_dtype)
        if self._rsv_comm_dtype.kind != 'f':
            raise ValueError('rsv_comm_dtype must be numpy.float16,'
                             'numpy.float32 or numpy.float64.')

        self._agv_comm_dtype = np.dtype(agv_comm_dtype)
        if self._agv_comm_dtype.kind != 'f':
            raise ValueError('agv_comm_dtype must be numpy.float16,'
                             'numpy.float32 or numpy.float64.')

        # GPU kernels. We don't generate here due to the same reason above
        self._cast_rsv_kernels = None
        self._cast_agv_kernels = None
        self._mean_kernel = None
        self._max_kernel = None
        self._memset_kernel = None

        # Packer to pack/unpack arrays
        self._packer = _utility.Packer()

        # For scaling in FP16
        self._scaled = False
        self._scaling_factors = None
        self._streams = None

    def _init_comms(self):
        if self.nccl_comm is not None:
            return

        if self.dims is None:
            self.nccl_comm = _communication_utility.init_nccl_comm(
                self.mpi_comm)
        else:
            if len(self.dims) == 0:
                self.nccl_comm = HierNcclCommunicator()
            else:
                self.nccl_comm = HierNcclCommunicator(self.dims)

    def reduce_scatterv_data(self, param_groups, extractors):
        """Executes Reduce+ScatterV.

            Flow(no cast): pack(A) -> send(A) -> recv(B) -> mean(B->A)
                                   -> unpack(A)

            Flow(casting): pack(C) -> cast(C->A) -> send(A) -> recv(B)
                                   -> mean(B->A) -> cast(A->C)
                                   -> unpack(C)
        """

        # CUDA default stream
        stream = chainer.cuda.Stream.null

        # Initialize NCCL communicator if not
        self._init_comms()
        # Target NCCL communicator
        nccl_comm = self.nccl_comm

        # This processes's assigned array index in arrays
        local_rank = self.rank

        # Extract arrays from param_groups
        arrays = _utility.extract(param_groups, self.indices, extractors)

        # Get total number of elements, local number of elements, and local
        # number of elements' offset
        nelems = _get_divideable_nelems(nccl_comm, _utility.get_nelems(arrays))
        nelems_local = _utility.get_nelems([arrays[local_rank]])
        nelems_offset = _utility.get_nelems(arrays[:local_rank])

        # Allocate memory if not
        needs_sync_a = _utility.assign(self.gpu_buf_a,
                                       nelems * self._rsv_comm_dtype.itemsize)
        needs_sync_b = _utility.assign(self.gpu_buf_b,
                                       nelems * self._rsv_comm_dtype.itemsize)
        needs_sync_c = _utility.assign(self.gpu_buf_c,
                                       nelems * self._arrs_dtype.itemsize) \
            if self._arrs_dtype != self._rsv_comm_dtype else False

        # Pack elements in a buffer
        # Data type casting will occur here if necessesary
        if self._arrs_dtype != self._rsv_comm_dtype:
            # Casting required
            if self._cast_rsv_kernels is None or \
                    self._cast_rsv_kernels.src_dtype != self._arrs_dtype or \
                    self._cast_rsv_kernels.dst_dtype != self._rsv_comm_dtype:
                self._cast_rsv_kernels = _CastingKernels(self._arrs_dtype,
                                                         self._rsv_comm_dtype)
            self._packcast(arrays, arrays, nelems, self.gpu_buf_c,
                           self.gpu_buf_a, self._cast_rsv_kernels, stream)
        else:
            # Casting unnecessesary
            self.packer.pack(arrays, self.gpu_buf_a, self._arrs_dtype.itemsize,
                             stream)

        # Buffers for AllReduce
        sendbuf = self.gpu_buf_a.ptr()
        recvbuf = self.gpu_buf_b.ptr()

        # Synchronize if necessesary
        if needs_sync_a or needs_sync_b or needs_sync_c:
            chainer.cuda.Stream.null.synchronize()

        # Communication
        nccl_dtype = _get_nccl_dtype(self._rsv_comm_dtype)
        nccl_comm.allReduce(sendbuf, recvbuf, nelems, nccl_dtype,
                            nccl.NCCL_SUM, stream.ptr)

        # Generate mean computing kernel if necessesary
        if self._mean_kernel is None:
            self._mean_kernel = _get_mean_kernel(self._rsv_comm_dtype,
                                                 self.size)

        # Compute the mean (divide by the number of processes)
        # TODO: compute mean and cast dtype simultaneously.
        self._mean_kernel(
            self.gpu_buf_b.array(
                nelems_local,
                offset=nelems_offset * self._rsv_comm_dtype.itemsize,
                dtype=self._rsv_comm_dtype),
            self.gpu_buf_a.array(
                nelems_local,
                offset=nelems_offset * self._rsv_comm_dtype.itemsize,
                dtype=self._rsv_comm_dtype),
            stream=stream)

        # Unpack elements from a buffer
        # Data type casting will occur here if necessesary
        if self._arrs_dtype != self._rsv_comm_dtype:
            # Casting required
            self._castunpack(arrays, [arrays[local_rank]], nelems_local,
                             self.gpu_buf_c, self.gpu_buf_a,
                             self._cast_rsv_kernels, stream,
                             offset=nelems_offset)
        else:
            # Casting unnecessesary
            self._packer.unpack([arrays[local_rank]], self.gpu_buf_a,
                                self._arrs_dtype.itemsize, stream,
                                offset=nelems_offset)

    def allgatherv_data(self, param_groups, extractors):
        """Executes AllGatherV.

            Flow(no cast): pack(A) -> send(A) -> recv(B) -> unpack(B)

            Flow(casting): pack(C) -> cast(C->A) -> send(A) -> recv(B)
                                   -> cast(B->C) -> unpack(C)
        """

        # CUDA default stream
        stream = chainer.cuda.Stream.null

        # This processes's assigned array index in arrays
        local_rank = self.rank

        # Allocate memory space for recieving
        # TODO
        #_utility.allocate_asgrad(param_groups, 'kfgrad')

        # Initialize NCCL communicator if not
        self._init_comms()

        # Target NCCL communicator
        nccl_comm = self.nccl_comm

        # Extract arrays from param_groups
        arrays = _utility.extract(param_groups, self.indices, extractors)

        # Get total number of elements, local number of elements, and local
        # number of elements' offset
        nelems = _get_divideable_nelems(nccl_comm, _utility.get_nelems(arrays))
        nelems_local = _utility.get_nelems([arrays[local_rank]])
        nelems_offset = _utility.get_nelems(arrays[:local_rank])

        # Allocate memory if not
        needs_sync_a = _utility.assign(self.gpu_buf_a,
                                       nelems * self._agv_comm_dtype.itemsize)
        needs_sync_b = _utility.assign(self.gpu_buf_b,
                                       nelems * self._agv_comm_dtype.itemsize)
        needs_sync_c = _utility.assign(self.gpu_buf_c,
                                       nelems * self._arrs_dtype.itemsize) \
            if self._arrs_dtype != self._agv_comm_dtype else False

        # Generate memset kernel if necessesary
        if self._memset_kernel is None:
            self._memset_kernel = _get_memset_kernel(self._agv_comm_dtype)

        # Memset 0
        self._memset_kernel(
            self.gpu_buf_a.array(nelems, dtype=self._agv_comm_dtype),
            stream=stream)

        # Pack elements in a buffer
        # Data type casting will occur here if necessesary
        if self._arrs_dtype != self._agv_comm_dtype:
            # Casting required
            if self._cast_agv_kernels is None or \
                    self._cast_agv_kernels.src_dtype != self._arrs_dtype or \
                    self._cast_agv_kernels.dst_dtype != self._agv_comm_dtype:
                self._cast_agv_kernels = _CastingKernels(self._arrs_dtype,
                                                         self._agv_comm_dtype)
            self._packcast(arrays, [arrays[local_rank]], nelems_local,
                           self.gpu_buf_c, self.gpu_buf_a,
                           self._cast_agv_kernels, stream,
                           offset=nelems_offset)
        else:
            # Casting unnecessesary
            self._packer.pack([arrays[local_rank]], self.gpu_buf_a,
                              self._arrs_dtype.itemsize,
                              stream, offset=nelems_offset)

        # Buffers for AllReduce
        sendbuf = self.gpu_buf_a.ptr()
        recvbuf = self.gpu_buf_b.ptr()

        # Synchronize if necessesary
        if needs_sync_a or needs_sync_b or needs_sync_c:
            chainer.cuda.Stream.null.synchronize()

        # Communication
        nccl_dtype = _get_nccl_dtype(self._agv_comm_dtype)
        nccl_comm.allReduce(sendbuf, recvbuf, nelems, nccl_dtype,
                            nccl.NCCL_SUM, stream.ptr)

        # Unpack elements from a buffer
        # Data type casting will occur here if necessesary
        if self._arrs_dtype != self._agv_comm_dtype:
            # Casting required
            self._castunpack(arrays, arrays, nelems, self.gpu_buf_c,
                             self.gpu_buf_b, self._cast_agv_kernels, stream,
                             offset=0)
        else:
            # Casting unnecessesary
            self._packer.unpack(arrays, self.gpu_buf_b,
                                self._arrs_dtype.itemsize, stream)

    def _packcast(self, global_arrays, arrays, nelems, src_gpu_buf,
                  dst_gpu_buf, casting_kernels, stream, offset=0):
        """Scale, pack, and cast using the given array and GPU buffers
        """

        # Scaling
        if casting_kernels.dst_dtype == np.dtype(np.float16):
            self._communication_scale(global_arrays, stream)

        # Pack elements to the buffer
        self.packer.pack(arrays, src_gpu_buf,
                         casting_kernels.src_dtype.itemsize, stream,
                         offset=offset)

        # Cast data type: SRC -> DST
        casting_kernels.src_to_dst_kernel(
            src_gpu_buf.array(
                nelems, dtype=casting_kernels.src_dtype,
                offset=offset * casting_kernels.src_dtype.itemsize),
            dst_gpu_buf.array(
                nelems, dtype=casting_kernels.dst_dtype,
                offset=offset * casting_kernels.dst_dtype.itemsize),
            stream=stream)

    def _castunpack(self, global_arrays, arrays, nelems, src_gpu_buf,
                    dst_gpu_buf, casting_kernels, stream, offset=0):
        """Cast, unpack, and scale using the given array and GPU buffers
        """

        # Cast data type: DST -> SRC
        casting_kernels.dst_to_src_kernel(
            dst_gpu_buf.array(
                nelems, dtype=casting_kernels.dst_dtype,
                offset=offset * casting_kernels.dst_dtype.itemsize),
            src_gpu_buf.array(
                nelems, dtype=casting_kernels.src_dtype,
                offset=offset * casting_kernels.src_dtype.itemsize),
            stream=stream)

        # Unpack elements from the buffer
        self.packer.unpack(arrays, src_gpu_buf,
                           casting_kernels.src_dtype.itemsize, stream,
                           offset=offset)

        # Scaling
        if self._scaled:
            self._rescale(global_arrays, stream)

    def _communication_scale(self, arrays, default_stream):
        if self._streams is None:
            self._streams = [cupy.cuda.Stream() for _ in range(NUM_STREAMS)]

        if self._max_kernel is None:
            self._max_kernel = _get_max_kernel()

        arrays = list(itertools.chain.from_iterable(arrays))
        arrays = [array for array, _ in arrays]
        arrays = sorted(arrays, key=lambda x: x.size)
        nelems = _get_divideable_nelems(self.nccl_comm, len(arrays))

        send_arr = cupy.empty(nelems, dtype=cupy.float32)
        recv_arr = cupy.empty(nelems, dtype=cupy.float32)

        used_stream_indices = np.zeros(NUM_STREAMS)
        for i, array in enumerate(arrays):
            stream_index = used_stream_indices.argmin()
            stream = self._streams[stream_index]

            self._max_kernel(array, send_arr[i], stream=stream)

            # NOTE(y1r): order of computation time is O(n).
            # So we count n (bin-packing problem heuristics)
            used_stream_indices[stream_index] += np.prod(array.shape)

        # NOTE(y1r): Assume that stream is default stream.
        # Therefore, stream.synchronize() is not necessary.
        self.nccl_comm.allReduce(send_arr.data.ptr,
                                 recv_arr.data.ptr,
                                 nelems,
                                 _get_nccl_dtype(cupy.dtype(cupy.float32)),
                                 nccl.NCCL_SUM,
                                 default_stream.ptr)

        with default_stream:
            scaling_factors = 65000 / recv_arr

            for scaling_factor, array in zip(scaling_factors, arrays):
                array *= scaling_factor

        self._scaled = True
        self._scaling_factors = scaling_factors

    def _rescale(self, arrays, default_stream):
        arrays = list(itertools.chain.from_iterable(arrays))
        arrays = [array for array, _ in arrays]
        arrays = sorted(arrays, key=lambda x: x.size)

        with default_stream:
            for i, array in enumerate(arrays):
                array *= (1 / self._scaling_factors[i])

        self._scaled = False


class _CastingKernels(object):

    def __init__(self, src_dtype, dst_dtype):
        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.src_to_dst_kernel = chainer.cuda.cupy.ElementwiseKernel(
            '{} x'.format(src_dtype.name),
            '{} y'.format(dst_dtype.name),
            'y = x', "{}_to_{}".format(src_dtype.name, dst_dtype.name))
        self.dst_to_src_kernel = chainer.cuda.cupy.ElementwiseKernel(
            '{} x'.format(dst_dtype.name),
            '{} y'.format(src_dtype.name),
            'y = x', "{}_to_{}".format(dst_dtype.name, src_dtype.name))


def _get_mean_kernel(dtype, size):
    return chainer.cuda.cupy.ElementwiseKernel(
        '{} x'.format(dtype.name),
        '{} y'.format(dtype.name),
        'y = x * (1.0 / {})'.format(size),
        'my_mean')


def _get_max_kernel():
    return chainer.cuda.cupy.ReductionKernel(
        'float32 x',
        'float32 y',
        'fabsf(x)',
        'fmaxf(a, b)',
        'y = a',
        '0',
        'my_max')


def _get_memset_kernel(dtype):
    return chainer.cuda.cupy.ElementwiseKernel(
        '',
        '{} x'.format(dtype.name),
        'x = 0.0',
        'my_memset')


def _get_divideable_nelems(nccl_comm, nelems):
    if hasattr(nccl_comm, 'getCountRequirement'):
        requirement = nccl_comm.getCountRequirement()
        return int(math.ceil(nelems / requirement)) * requirement
    else:
        return nelems


def _get_nccl_dtype(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    else:
        raise ValueError(
            'dtype must be numpy.float16, numpy.float32 or numpy.float64,'
            'not {}'.format(dtype))
