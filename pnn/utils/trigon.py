import numpy as np


def atan2f(y, x, tolerance=1e-6, to_degree=False):
    zero_y = np.abs(y) <= tolerance
    zero_x = np.abs(x) <= tolerance
    if zero_x and zero_y:
        rad = 0
    elif zero_x and (not zero_y):
        rad = np.pi/2 if y > tolerance else -np.pi/2
    elif (not zero_x) and zero_y:
        rad = 0 if x > tolerance else np.pi
    else:
        rad = np.arctan2(y, x)
    if to_degree:
        return np.rad2deg(rad)
    else:
        return rad
    

def angle_diff(comp_src, comp_dst, offset=0, tolerance=1e-6, wrap=True, to_degree=False):
    zero_src = np.abs(comp_src) <= tolerance
    zero_dst = np.abs(comp_dst) <= tolerance
    if zero_src and zero_dst:
        rad = 0
    elif zero_src and (not zero_dst):
        rad = np.angle(comp_dst)
    elif (not zero_src) and zero_dst:
        rad = -np.angle(comp_src)
    else:
        rad = np.angle(comp_dst) - np.angle(comp_src)
    rad += offset
    if wrap:
        rad = np.mod(rad, 2 * np.pi)
    if to_degree:
        return np.rad2deg(rad)
    else:
        return rad