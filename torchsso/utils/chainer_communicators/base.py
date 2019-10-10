from chainermn.communicators import mpi_communicator_base
import warnings

from torchsso.utils.chainer_communicators import _utility


class KFACCommunicatorBase(mpi_communicator_base.MpiCommunicatorBase):

    def __init__(self, mpi_comm):
        super(KFACCommunicatorBase, self).__init__(mpi_comm)
        self.indices = None
        self.packer = _utility.Packer()

    def allreduce_grad(self):
        # We don't use AllReduce for training K-FAC
        warnings.warn('AllReduce called, skipping...')

    def reduce_scatterv_data(self, fblocks, extractors):
        raise NotImplementedError

    def allgatherv_data(self, fblocks, extractors):
        raise NotImplementedError
