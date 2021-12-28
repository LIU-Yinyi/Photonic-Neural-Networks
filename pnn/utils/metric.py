import numpy as np


def fidelity(U, Ue):
    assert U.ndim == 2 and Ue.ndim == 2
    assert len(U) == len(Ue)
    dim = len(U)
    return np.abs(np.trace(U.conj().T @ Ue) / np.sqrt(dim * np.trace(Ue.conj().T @ Ue))) ** 2
