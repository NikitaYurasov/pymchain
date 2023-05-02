import numpy as np

from tqdm import tqdm
from loguru import logger
from typing import Union, List
from utils import random_stochastic_vector, generate_multinomial_rv


class MarkovChain:
    """
    A Markov chain generator of arbitrary depth. By default, a homogeneous Markov chain is used, that is, the transition
    tensor does not depend on the iteration number.
    There are two use cases:
        1. Creating a random transition tensor based on a polynomial distribution using input frequencies;
        2. Setting the tensor during initialization.
    An important condition is the stochastic tensor property!
    Parameters
    ----------
    depth : int
        the depth of the process dependency (must be >= 1)
    input_p : np.ndarray | List
        probability vector for selecting the first elements
    transition_pvals : np.ndarray | List, optional
        probability vector for generating a random tensor
    transition_tensor : np.ndarray | List, optional
        transition Tensor
    prng : optional
        PRNG object
    prng_seed : optional
        seed for default deterministic PRNG
    iterations_for_tensor : int
        the number of iterations when generating a sample from a polynomial distribution
    alphabet : np.ndarray | List, optional
        symbols substituted instead of generated elements
    """

    def __init__(
        self,
        depth: int,
        input_p: Union[np.ndarray, List],
        transition_pvals: Union[np.ndarray, List] = None,
        transition_tensor: Union[np.ndarray, List] = None,
        iterations_for_tensor: int = 1000,
        alphabet: Union[np.ndarray, List] = None,
        prng=None,
        prng_seed=None,
    ):
        if depth < 1 or isinstance(depth, float):
            raise ValueError(f'The depth attribute should be integer and >= 1, got {depth}')
        if not isinstance(input_p, (list, np.ndarray)):
            raise TypeError(f'Type of input vector must be <list> or <np.ndarray>, got {type(input_p)}')
        if isinstance(input_p, list):
            input_p = np.array(input_p)

        self.depth = depth
        self.input_p = input_p
        self.n = len(input_p)
        self._transitions_pvals = transition_pvals
        self._prng = prng or np.random.default_rng(seed=prng_seed)
        if iterations_for_tensor < 1 or isinstance(iterations_for_tensor, float):
            raise ValueError(f'Number of iterations must be integer and >= 1, got {iterations_for_tensor}')
        self.iter_for_tensor = iterations_for_tensor

        self.transition_tensor = None

        if transition_tensor is None:
            if transition_pvals is None:
                logger.warning('Transition pvals will be generated from polynomial distribution')
                transition_pvals = random_stochastic_vector(len(input_p), prng=self._prng)
            if len(input_p) != len(transition_pvals):
                raise ValueError(
                    f'Mismatch at input vector and transition pvals, got {len(input_p)} and '
                    f'{len(transition_pvals)} lengths corresponding'
                )
            self._generate_transition_tensor()
        else:
            self._set_transition_tensor(transition_tensor)

        if alphabet is not None and len(alphabet) != len(input_p):
            raise ValueError(
                f"Lengths of initial pvals and passed symbols must be equal, "
                f"got {len(input_p)} and {len(alphabet)} instead"
            )
        self.symbols = alphabet

        self.sequence = None

    def _set_transition_tensor(self, tensor: Union[np.ndarray, List], rel_tol: float = 1e-6):
        """
        The function of checking and setting the transition tensor, if such was specified during initialization.
        Parameters
        ----------
        tensor : np.ndarray | List, optional
            transition Tensor
        rel_tol : float
            maximum deviation of machine accuracy when testing for convergence to 1
        """
        if not isinstance(tensor, np.ndarray) and isinstance(tensor, list):
            tensor = np.array(tensor)
        if tensor.ndim - 1 != self.depth:
            raise ValueError(
                f'Mismatch at number of dims in tensor and depth, got {tensor.ndim} and {self.depth} corresponding'
            )
        if tensor.size != tensor.shape[0] ** tensor.ndim:
            raise ValueError(f'Tensor dimensions must be equal, got {tensor.shape}')

        index_3d = np.zeros(self.depth + 1, dtype=np.uint32)
        for i in range(self.n**self.depth):
            _sum = 0.0
            for j in range(self.n):
                _sum += tensor.item(tuple(index_3d))
                index_3d = self._update_3d_index(index_3d)
            if np.abs(1.0 - _sum) > rel_tol:
                raise ValueError(f'Sum: {_sum} does not coverage not to 1\nSee before {index_3d} index')
        self.transition_tensor = tensor

    def _update_3d_index(self, index):
        """
        Updating an arbitrary-sized tensor index in the correct order
        Parameters
        ----------
        index : List, np.ndarray
            3d-index

        Returns
        -------
        List, np.ndarray
            updated 3d-index
        """
        for i in range(len(index) - 1, -1, -1):
            if index[i] == self.n - 1:
                index[i] = 0
                continue
            else:
                index[i] += 1
                break
        return index

    def _generate_transition_tensor(self):
        """
        Creating a transition tensor randomly
        """
        matrix = np.zeros([len(self.input_p)] * (self.depth + 1), dtype=np.float64)
        index_3d = [0] * (self.depth + 1)
        for _ in tqdm(range(self.n**self.depth), desc='Transition Tensor Generation'):
            rand_perm = generate_multinomial_rv(self.iter_for_tensor, self._transitions_pvals) / self.iter_for_tensor
            self._prng.shuffle(rand_perm)
            for j in range(self.n):
                matrix.itemset(tuple(index_3d), rand_perm[j])
                index_3d = self._update_3d_index(index_3d)
        self.transition_tensor = matrix

    @staticmethod
    def _check_u_in_interval(u, left_bound, right_bound, is_first_interval=True):
        """
        Checks whether a uniformly distributed random variable belongs to an interval depending on the position
        of the interval.
        For more information, see Ivchenko G.I., Medvedev Yu.I. - "Mathematical statistics" section about modeling
        of polynomial distribution and Markov chains.
        Parameters
        ----------
        u : float
            random variable from U[0, 1]
        left_bound : float
            the left bound of the interval
        right_bound : float
            the right bound of the interval
        is_first_interval : bool
            flag; only in first interval left bound has to be included

        Returns
        -------
        bool
            is passed rv in interval
        """
        if is_first_interval:
            return left_bound <= u <= right_bound
        else:
            return left_bound < u <= right_bound

    def _generate_start_transition(self):
        """
        Creating the first states of the chain. The number of initial states depends on the depth of dependence
        Returns
        -------
        np.ndarray
            array of initial states
        """
        start_transitions = np.empty(self.depth, dtype=np.uint64)
        intervals = np.append(np.array([0]), np.cumsum(self.input_p))
        for i in range(self.depth):
            u = self._prng.random()
            first_iter = True
            for j in range(intervals.shape[0] - 1):
                if j != 0:
                    first_iter = False
                if self._check_u_in_interval(u, intervals[j], intervals[j + 1], is_first_interval=first_iter):
                    start_transitions[i] = j
                    break
        return start_transitions

    def _generate_transition(self, prev_transitions: np.ndarray):
        """
        Simulation of a new state of the circuit depending on the previous ones.
        Parameters
        ----------
        prev_transitions : np.ndarray
            array of previous states

        Returns
        -------
        int
            new state
        """
        vector = self.transition_tensor
        for i in range(prev_transitions.shape[0]):
            vector = vector[prev_transitions[i]]
        intervals = np.append(np.array([0]), np.cumsum(vector))
        u = self._prng.random()
        first_iter = True
        new_transition = -1
        for j in range(intervals.shape[0] - 1):
            if j != 0:
                first_iter = False
            if self._check_u_in_interval(u, intervals[j], intervals[j + 1], is_first_interval=first_iter):
                new_transition = j
                break
        if new_transition == -1:
            raise AssertionError(f'While generating new transition came vector {vector}.\nCheck your transition tensor')
        return new_transition

    def generate_chain(self, steps: int):
        """
        Simulations of the entire Markov chain
        Parameters
        ----------
        steps : int
            number of simulated circuit states
        Returns
        -------
        np.ndarray
            array of chain states
        """
        if isinstance(steps, float):
            logger.warning('Be careful, type of steps should be <int>. Steps will be transformed into <int>')
        try:
            steps = int(steps)
        except TypeError:
            raise TypeError(f'Expected type of steps is <int>, got {type(steps)}')
        seq = np.empty(steps, dtype=np.uint64)
        seq[0 : self.depth] = self._generate_start_transition()
        for i in tqdm(range(self.depth, steps), desc='Chain Generation'):
            seq[i] = self._generate_transition(seq[i - self.depth : i])
        if self.symbols is not None:
            old_seq = seq.copy()
            seq = np.empty_like(old_seq, dtype=object)
            for i in range(old_seq.size):
                seq[i] = self.symbols[old_seq[i]]
        self.sequence = seq
        return seq
