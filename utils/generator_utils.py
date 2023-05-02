import numpy as np
from typing import Union
from numba import jit, prange


def random_stochastic_vector(n: int, left_bound: int = 0, right_bound: int = 999, prng=None):
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
    prng : optional
        PRNG

    Returns
    -------
    np.ndarray
        stochastic vector
    """
    random_int_vec = np.zeros(n)
    for i in range(n):
        random_int_vec[i] = prng.integers(left_bound, right_bound)
    _sum = random_int_vec.sum()
    return random_int_vec / _sum


@jit(['uint64[:](uint64, float64[:])'])
def generate_multinomial_rv(n: int, p_vec: Union[np.ndarray, list]):
    """
    A function for modeling a sample of size n from a polynomial distribution.
    For more information, see Ivchenko G.I., Medvedev Yu.I. - "Mathematical statistics" paragraph about modeling
    of a polynomial distribution.
    Parameters
    ----------
    n : int
        sample size
    p_vec : np.ndarray | List
        probability vector

    Returns
    -------
    np.ndarray
        array of sorted selections by elements
    """
    seq = np.zeros(p_vec.shape[0], dtype=np.uint64)
    cum_sum = np.append(np.array([0]), np.cumsum(p_vec))
    for _ in prange(n):
        u = np.random.random()
        for j in range(cum_sum.shape[0] - 1):
            if j == 0 and cum_sum[j] <= u <= cum_sum[j + 1]:
                seq[0] += 1
                break
            elif cum_sum[j] < u <= cum_sum[j + 1]:
                seq[j] += 1
                break
    return seq
