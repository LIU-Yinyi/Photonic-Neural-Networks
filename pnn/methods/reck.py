import numpy as np
import sympy as sp

from pnn.utils import angle_diff, atan2f
from pnn.utils import U2BS, U2MZI


def decompose_reck(u, block='bs'):
    assert isinstance(u, np.ndarray)
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    if len(u.shape) != 2:
        raise ValueError("U(N) should be 2-dimension matrix.")
        
    if u.shape[0] != u.shape[1]:
        raise ValueError("U(N) should be a square matrix.")
        
    mat = u.copy().astype(np.complex128)
    dim = mat.shape[0]
    num = int(dim * (dim - 1) / 2)
    phis = np.zeros(num)
    thetas = np.zeros(num)
    alphas = np.zeros(dim)
    index = 0
    for p in range(1, dim):
        x = dim - p
        for q in range(dim-p, 0, -1):
            y = q - 1
            if block == 'bs':
                thetas[index] = atan2f(np.abs(mat[x,y]), np.abs(mat[x,x]))
                phis[index] = angle_diff(mat[x,x], mat[x,y], offset=-np.pi/2)
                U2block = U2BS
            elif block == 'mzi':
                thetas[index] = np.pi/2 - atan2f(np.abs(mat[x,y]), np.abs(mat[x,x]))
                phis[index] = angle_diff(mat[x,x], mat[x,y], offset=np.pi)
                U2block = U2MZI
            mat = mat @ U2block(dim, y, x, phis[index], thetas[index]).conj().T
            index += 1
    for i in range(dim):
        alphas[i] = np.angle(mat[i, i])
    return phis, thetas, alphas


def reconstruct_reck(phis, thetas, alphas, block='bs', Lp_dB=0, Lc_dB=0):
    assert len(phis.squeeze().shape) == 1
    assert len(thetas.squeeze().shape) == 1
    assert len(alphas.squeeze().shape) == 1
    assert phis.squeeze().shape[0] == thetas.squeeze().shape[0]
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    if block == 'bs':
        U2block = U2BS
    elif block == 'mzi':
        U2block = U2MZI
    
    num = thetas.squeeze().shape[0]
    dim = int((1 + np.sqrt(1 + 8 * num))/ 2)
    assert alphas.squeeze().shape[0] == dim
    
    Lp = 10 ** (Lp_dB / 10)
    Lc = 10 ** (Lc_dB / 10)
    
    mat = np.diag(np.exp(1j * alphas))
    index = num
    for p in range(1, dim):
        for q in range(p):
            index -= 1
            mat = mat @ U2block(dim, q, p, phis[index], thetas[index], Lp=Lp, Lc=Lc)
    return mat