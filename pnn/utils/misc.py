from scipy.stats import unitary_group


def unitary(dim: int):
    return unitary_group.rvs(dim)