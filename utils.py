import numpy as np

def uniform_indices(total: int, T: int):
    if total <= 0:
        return [0]*T
    if total >= T:
        return np.linspace(0, total-1, num=T, dtype=np.int32).tolist()
    idx = np.linspace(0, total-1, num=T)
    return np.clip(np.round(idx).astype(np.int32), 0, max(0, total-1)).tolist()
