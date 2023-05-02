import numpy as np


def random_stochastic_vector(n: int, left_bound: int = 0, right_bound: int = 999, seed=None):
    """
    Creating a stochastic vector from a polynomial distribution with the number of elements <n>.
    Parameters
    ----------
    n : int
        number of elements for a polynomial scheme
    left_bound : int
        left bound of precision
    right_bound : int
        right bound of precision
    seed : optional
        PRNG seed

    Returns
    -------
    np.ndarray
        stochastic vector
    """
    rng = np.random.default_rng(seed=seed)
    random_int_vec = np.zeros(n)
    for i in range(n):
        random_int_vec[i] = rng.integers(left_bound, right_bound)
    _sum = random_int_vec.sum()
    return random_int_vec / _sum
