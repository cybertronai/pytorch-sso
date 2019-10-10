try:
    import cupy
except:
    # print("No cupy detected")
    pass

from torch.utils.dlpack import to_dlpack, from_dlpack


def to_cupy(m_tensor):
    return cupy.fromDlpack(to_dlpack(m_tensor))


def from_cupy(m_cp):
    return from_dlpack(m_cp.toDlpack())
