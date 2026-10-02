import numpy as np
def read212(path):
    raw = np.fromfile(path, np.uint8).reshape(-1, 3).astype(np.int32)
    a = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    b = raw[:, 2] | ((raw[:, 1] & 0xF0) << 4)
    a[a > 2047] -= 4096; b[b > 2047] -= 4096
    return np.stack([a, b], 1).astype(float)
