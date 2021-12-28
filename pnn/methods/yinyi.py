import numpy as np
from pnn import methods
import sympy as sp

from pnn.utils import angle_diff, atan2f
from pnn.utils import U2BS, U2MZI
from pnn.utils import cossin

from pnn.methods import decompose_reck, reconstruct_reck
from pnn.methods import decompose_clements, reconstruct_clements


class UmiCsd:
    def __init__(self, matrix=None, p=None, q=None):
        self.u1 = None
        self.u2 = None
        self.theta = None
        self.v1h = None
        self.v2h = None
        self.matrix = None
        
        if matrix is not None:
            self.decompose(matrix, p, q)
    
    def __repr__(self):
        return "{}".format(self.matrix)
    
    def form_cs(self, Lp_dB=0, Lc_dB=0, swap_sign=False):
        cs = []
        Lp = 10 ** (Lp_dB / 10)
        Lc = 10 ** (Lc_dB / 10)
        for t in self.theta:
            tmp = np.array([
                [np.sqrt(Lp) * np.cos(t), -np.sqrt(Lc) * np.sin(t)],
                [np.sqrt(Lc) * np.sin(t), np.sqrt(Lp) * np.cos(t)]
            ])
            tmp = tmp.T if swap_sign else tmp
            cs.append(tmp)
        return cs
    
    @staticmethod
    def rearrange_vector(vec1, vec2):
        assert len(vec1) == len(vec2)
        assert len(vec1) == vec1.size
        assert len(vec2) == vec2.size
        
        N = len(vec1)
        retval = []
        for i in range(N):
            tmp = np.concatenate((vec1[i], vec2[i])).reshape(2, 1)
            retval.append(tmp)
            
        return retval
    
    def decompose(self, matrix, p=None, q=None):
        assert matrix.ndim == 2, "[ERROR] Dimension of Input Matrix is NOT 2."
        assert np.allclose(matrix @ matrix.conj().T, np.eye(len(matrix))), "[ERROR] Input Matrix is NOT Unitary."
        
        if p is None and q is None:
            p = len(matrix) // 2
            q = len(matrix) - p
        elif p is None:
            p = len(matrix) - q
        elif q is None:
            q = len(matrix) - p
            
        self.matrix = matrix
        (self.u1, self.u2), self.theta, (self.v1h, self.v2h) = cossin(matrix, p=p, q=q, separate=True)
    
    def decompose_recursive(self, depth):
        def check_and_csd(mat, d):
            if isinstance(mat, np.ndarray):
                if len(mat) <= 2:
                    return mat
                if d > 0:
                    ret = UmiCsd(mat)
                    ret.decompose_recursive(d-1)
                    return ret
                else:
                    return UmiCsd(mat)
            elif isinstance(mat, UmiCsd):
                return mat
            else:
                raise "[ERROR] Not Supported Type in func<decompose_recursive>."
        
        self.u1 = check_and_csd(self.u1, depth-1)
        self.u2 = check_and_csd(self.u2, depth-1)
        self.v1h = check_and_csd(self.v1h, depth-1)
        self.v2h = check_and_csd(self.v2h, depth-1)
        
    def reconstruct(self, Lp_dB=0, Lc_dB=0, method='clements', block='bs'):
        assert method.lower() in ['reck', 'clements']
        assert block.lower() in ['bs', 'mzi']
        if method.lower() == 'reck':
            planar_decompose = decompose_reck
            planar_reconstruct = reconstruct_reck
        elif method.lower() == 'clements':
            planar_decompose = decompose_clements
            planar_reconstruct = reconstruct_clements
        
        def check_and_rcs(mat):
            if isinstance(mat, np.ndarray):
                if len(mat) < 2:
                    return mat
                [p, t, a] = planar_decompose(mat, block=block.lower())
                return planar_reconstruct(p, t, a, block=block.lower(), Lp_dB=Lp_dB, Lc_dB=Lc_dB)
            elif isinstance(mat, UmiCsd):
                return mat.reconstruct(Lp_dB=Lp_dB, Lc_dB=Lc_dB, method=method, block=block)
            else:
                raise "[Error] Not Supported Type in func<reconstruct>."
                
        _u1 = check_and_rcs(self.u1)
        _u2 = check_and_rcs(self.u2)
        _v1h = check_and_rcs(self.v1h)
        _v2h = check_and_rcs(self.v2h)
        
        Lp = 10 ** (Lp_dB / 10)
        Lc = 10 ** (Lc_dB / 10)
        
        def bridge_matrix(factor, value, cs, m1, m2):
            assert cs in ['sin', 'cos']
            if cs == 'sin':
                ops = np.sin
            elif cs == 'cos':
                ops = np.cos
            
            m = np.eye(m1.shape[1] if m1.shape[1] >= m2.shape[0] else m2.shape[0])
            l = len(value)
            m[:l, :l] = factor * np.diag(ops(value))
            return m[:m1.shape[1], :m2.shape[0]]
        
        _cs11 = bridge_matrix(np.sqrt(Lp), self.theta, 'cos', _u1, _v1h)
        _cs12 = bridge_matrix(np.sqrt(Lc), -self.theta, 'sin', _u1, _v2h)
        _cs21 = bridge_matrix(np.sqrt(Lc), self.theta, 'sin', _u2, _v1h)
        _cs22 = bridge_matrix(np.sqrt(Lp), self.theta, 'cos', _u2, _v2h)
        
        _b11 = _u1 @ _cs11 @ _v1h
        _b12 = _u1 @ _cs12 @ _v2h
        _b21 = _u2 @ _cs21 @ _v1h
        _b22 = _u2 @ _cs22 @ _v2h
            
        return np.block([[_b11, _b12], [_b21, _b22]])


def decompose_yinyi(u, block='', p=None, q=None, depth=0):
    umi = UmiCsd(u, p, q)
    if depth > 0:
        umi.decompose_recursive(depth)
    return umi


def reconstruct_yinyi(umi, Lp_dB=0, Lc_dB=0, method='clements', block='bs'):
    assert isinstance(umi, UmiCsd)
    return umi.reconstruct(Lp_dB=Lp_dB, Lc_dB=Lc_dB, method=method, block=block)
