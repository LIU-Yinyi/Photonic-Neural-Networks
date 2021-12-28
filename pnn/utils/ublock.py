import sympy as sp
import numpy as np


def U2BS(dim, m, n, phi, theta, use_sym=False, Lp=1, Lc=1):
    assert m < n < dim
    if use_sym:
        mat = sp.eye(dim)
        mat[m, m] = sp.sqrt(Lp) * sp.exp(sp.I * phi) * sp.cos(theta)
        mat[m, n] = sp.sqrt(Lc) * sp.I * sp.sin(theta)
        mat[n, m] = sp.sqrt(Lc) * sp.I * sp.exp(sp.I * phi) * sp.sin(theta)
        mat[n, n] = sp.sqrt(Lp) * sp.cos(theta)
    else:
        mat = np.eye(dim, dtype=np.complex128)
        mat[m, m] = np.sqrt(Lp) * np.exp(1j * phi) * np.cos(theta)
        mat[m, n] = np.sqrt(Lc) * 1j * np.sin(theta)
        mat[n, m] = np.sqrt(Lc) * 1j * np.exp(1j * phi) * np.sin(theta)
        mat[n, n] = np.sqrt(Lp) * np.cos(theta)
    return mat


def U2MZI(dim, m, n, phi, theta, use_sym=False, Lp=1, Lc=1):
    assert m < n < dim
    if use_sym:
        mat = sp.eye(dim)
        mat[m, m] = sp.sqrt(Lp) * sp.I * sp.exp(sp.I * phi) * sp.sin(theta)
        mat[m, n] = sp.sqrt(Lc) * sp.I * sp.cos(theta)
        mat[n, m] = sp.sqrt(Lc) * sp.I * sp.exp(sp.I * phi) * sp.cos(theta)
        mat[n, n] = -sp.sqrt(Lp) * sp.I * sp.sin(theta)
    else:
        mat = np.eye(dim, dtype=np.complex128)
        mat[m, m] = np.sqrt(Lp) * 1j * np.exp(1j * phi) * np.sin(theta)
        mat[m, n] = np.sqrt(Lc) * 1j * np.cos(theta)
        mat[n, m] = np.sqrt(Lc) * 1j * np.exp(1j * phi) * np.cos(theta)
        mat[n, n] = -np.sqrt(Lp) * 1j * np.sin(theta)
    return mat
