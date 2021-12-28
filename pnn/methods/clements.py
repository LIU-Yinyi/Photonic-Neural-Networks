import numpy as np
import sympy as sp

from pnn.utils import angle_diff, atan2f
from pnn.utils import U2BS, U2MZI


def decompose_clements(u, block='bs'):
    assert isinstance(u, np.ndarray)
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    if len(u.shape) != 2:
        raise ValueError("U(N) should be 2-dimension matrix.")
        
    if u.shape[0] != u.shape[1]:
        raise ValueError("U(N) should be a square matrix.")
        
    mat = u.copy().astype(np.complex128)
    dim = mat.shape[0]
    
    row = dim - 1
    col = int(np.ceil(dim / 2))
    
    cnt_fore = np.zeros(row, dtype=int)
    cnt_back = np.ones(row, dtype=int) * (col - 1)
    if dim % 2 == 1:
        cnt_back[1::2] = col - 2
    
    phis = np.zeros((row, col))
    thetas = np.zeros((row, col))
    alphas = np.zeros(dim)
    
    for p in range(dim-1):
        for q in range(p+1):
            if p % 2 == 0:
                x = dim - 1 - q
                y = p - q
                if block == 'bs':
                    theta = atan2f(np.abs(mat[x,y]), np.abs(mat[x,y+1]))
                    phi = angle_diff(mat[x,y+1], mat[x,y], offset=-np.pi/2)
                    U2block = U2BS
                elif block == 'mzi':
                    theta = np.pi/2 - atan2f(np.abs(mat[x,y]), np.abs(mat[x,y+1]))
                    phi = angle_diff(mat[x,y+1], mat[x,y], offset=np.pi)
                    U2block = U2MZI
                mat = mat @ U2block(dim, y, y+1, phi, theta).conj().T
                thetas[y, cnt_fore[y]] = theta
                phis[y, cnt_fore[y]] = phi
                cnt_fore[y] += 1
            else:
                x = dim - 1 - p + q
                y = q
                if block == 'bs':
                    theta = atan2f(np.abs(mat[x,y]), np.abs(mat[x-1,y]))
                    phi = angle_diff(mat[x-1,y], mat[x,y], offset=np.pi/2)
                    U2block = U2BS
                elif block == 'mzi':
                    theta = np.pi/2 - atan2f(np.abs(mat[x,y]), np.abs(mat[x-1,y]))
                    phi = angle_diff(mat[x-1,y], mat[x,y], offset=0)
                    U2block = U2MZI
                mat = U2block(dim, x-1, x, phi, theta) @ mat
                thetas[x-1, cnt_back[x-1]] = theta
                phis[x-1, cnt_back[x-1]] = phi
                cnt_back[x-1] -= 1
    for p in range(dim-2, -1, -1):
        for q in range(p, -1, -1):
            if p % 2 == 0:
                continue
            x = dim - 1 - p + q
            y = q
            cnt_back[x-1] += 1
            theta = thetas[x-1, cnt_back[x-1]]
            phi = phis[x-1, cnt_back[x-1]]
            eta1 = mat[x-1, x-1]
            eta2 = mat[x, x]
            if block == 'bs':
                phi_new = angle_diff(eta2, -eta1, offset=0)
                mat[x-1, x-1] = eta1 * np.exp(-1j * (phi+phi_new))
            elif block == 'mzi':
                phi_new = angle_diff(eta2, eta1, offset=0)
                mat[x-1, x-1] = -eta1 * np.exp(-1j * (phi+phi_new))
                mat[x, x] = -eta2
            phis[x-1, cnt_back[x-1]] = phi_new
    for i in range(dim):
        alphas[i] = np.angle(mat[i, i])
    return phis, thetas, alphas


def reconstruct_clements(phis, thetas, alphas, block='bs', Lp_dB=0, Lc_dB=0):
    assert len(phis.squeeze().shape) == 2 or phis.size == 1
    assert len(thetas.squeeze().shape) == 2 or thetas.size == 1
    assert len(alphas.squeeze().shape) == 1
    assert phis.squeeze().shape == thetas.squeeze().shape
    assert isinstance(block, str) and block.strip().lower() in ['bs', 'mzi']
    
    if block == 'bs':
        U2block = U2BS
    elif block == 'mzi':
        U2block = U2MZI
    
    if thetas.size == 1:
        row = 1
        col = 1
    else:
        row, col = thetas.squeeze().shape
    dim = row + 1
    num = int(dim * (dim - 1) / 2) 
    assert alphas.squeeze().shape[0] == dim
    
    Lp = 10 ** (Lp_dB / 10)
    Lc = 10 ** (Lc_dB / 10)
    
    sft = np.diag(np.exp(1j * alphas))
    mat = np.eye(dim)
    for p in range(col):
        for q in range(0, row, 2):
            mat = U2block(dim, q, q+1, phis[q,p], thetas[q,p], Lp=Lp, Lc=Lc) @ mat
        if p >= col - 1 and dim % 2 == 1:
            continue
        for q in range(1, row, 2):
            mat = U2block(dim, q, q+1, phis[q,p], thetas[q,p], Lp=Lp, Lc=Lc) @ mat
    mat = sft @ mat
    return mat
    