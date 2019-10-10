import numpy as np


def create_communicator(communicator_name='pure_nccl',
                        mpi_comm=None,
                        rsv_comm_dtype=np.float32,
                        agv_comm_dtype=np.float32,
                        use_hiercoll=False,
                        dims=None,
                        ):
    if mpi_comm is None:
        import mpi4py.MPI
        mpi_comm = mpi4py.MPI.COMM_WORLD

    if communicator_name != 'pure_nccl' and rsv_comm_dtype != np.float32:
        raise ValueError(
            'rsv_comm_dtype is only available at \'pure_nccl\' communicator')

    if communicator_name != 'pure_nccl' and agv_comm_dtype != np.float32:
        raise ValueError(
            'agv_comm_dtype is only available at \'pure_nccl\' communicator')

    if communicator_name != 'pure_nccl' and dims is not None:
        raise ValueError(
            'dims is only available at \'pure_nccl\' communicator')

    if communicator_name == 'pure_nccl':
        from torchsso.utils.chainer_communicators.pure_nccl_communicator \
            import PureNCCLCommunicator
        return PureNCCLCommunicator(mpi_comm,
                                    rsv_comm_dtype=rsv_comm_dtype,
                                    agv_comm_dtype=agv_comm_dtype,
                                    use_hiercoll=use_hiercoll,
                                    dims=dims
                                    )
    else:
        raise ValueError(
            'Unrecognized communicator_name: {}'.format(communicator_name))
