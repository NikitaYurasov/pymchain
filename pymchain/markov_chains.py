import numpy as np
from tqdm import tqdm
from numba import jit, prange
from utils.generator_utils import random_stochastic_vector


@jit(['uint64[:](uint64, float64[:])'])
def generate_multinomial_rv(n, p_vec):
    """
    Функция моделирования выборки размера n из полиномиального распределения.
    Для большего понимания см. Ивченко Г.И., Медведев Ю.И. - "Математическая статистика" параграф про моделирование
    полиномиального распределение.
    Parameters
    ----------
    n : int
        Размер выборки
    p_vec : Union[np.ndarray, list]
        Вектор вероятностей

    Returns
    -------
    np.ndarray
        Массив отсортированной выборки по элементам
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


class MarkovChain:
    def __init__(self,
                 depth,
                 input_p,
                 transition_pvals=None,
                 transition_tensor=None,
                 seed=None,
                 iterations_for_tensor=1000,
                 symbols=None):
        """
        Генератор цепи Маркова произвольной глубины. По умолчанию используется однородная цепь Маркова, то есть
        тензор переходов не зависит от номера итерации.
        Есть два варианта использования:
            1. Создание случайного тезора переходов на основе полиномиального распределения, используя входные частоты;
            2. Задание тензора при инициализации.
        Важным условием является свойство стохастического тензора!
        Parameters
        ----------
        depth : int
            Глубина зависимости процесса (должна быть >= 1)
        input_p : Union[np.ndarray, list]
            Вектор вероятностей для выбора первых элементов
        transition_pvals : Union[np.ndarray, list, None], optional
            Вектор вероятностей для генерации случайного тензора
        transition_tensor : Union[np.ndarray, list, None], optional
            Тензор переходов
        seed : optional
            Зерно для ГПСЧ
        iterations_for_tensor : int
            Количество итераций при генерации выборки из полиномиального распределения
        symbols : Union[np.ndarray, list, None], optional
            Символы, подставляемые вместо генерируемых элементов
        """
        if depth < 1 or isinstance(depth, float):
            raise AttributeError(f'The depth attribute should be integer and >= 1, got {depth}')
        if not isinstance(input_p, (list, np.ndarray)):
            raise TypeError(f'Type of input vector must be <list> or <np.ndarray>, got {type(input_p)}')
        if isinstance(input_p, list):
            input_p = np.array(input_p)
        self.depth = depth
        self.input_p = input_p
        self.n = len(input_p)
        self._transitions_pvals = transition_pvals
        self._rng = np.random.default_rng(seed=seed)
        if iterations_for_tensor < 1 or isinstance(iterations_for_tensor, float):
            raise AttributeError(f'Number of iterations must be integer and >= 1, got {iterations_for_tensor}')
        self.iter_for_tensor = iterations_for_tensor

        self.transition_tensor = None

        if transition_tensor is None:
            if transition_pvals is None:
                print('WARNING: transition pvals will be generated from polynomial distribution')
                transition_pvals = random_stochastic_vector(len(input_p), seed=seed)
            if len(input_p) != len(transition_pvals):
                raise AttributeError(f'Mismatch at input vector and transition pvals, got {len(input_p)} and '
                                     f'{len(transition_pvals)} lengths corresponding')
            self._generate_transition_tensor()
        else:
            self._set_transition_tensor(transition_tensor)

        if len(symbols) != len(input_p) and symbols is not None:
            raise AttributeError(f"Lengths of initial pvals and passed symbols must be equal, "
                                 f"got {len(input_p)} and {len(symbols)} instead")
        self.symbols = symbols

        self.sequence = None

    def get_block_sequence(self, block_len=64):
        """
        Функция возвращает преобразованную последовательность в блоках с заданной длиной <block_len>
        Parameters
        ----------
        block_len : int
            Длина блока

        Returns
        -------
        np.ndarray
            Двочиная последовательность
        """
        array = np.zeros(self.sequence.size * block_len, dtype=np.uint8)
        for i in range(self.sequence.size):
            block = array[block_len * i: block_len * i + block_len]
            num = self.sequence[i]
            p = block_len - 1
            while num > 0:
                block[p] = num % 2
                num //= 2
                p -= 1
        return array

    def get_int_sequence(self):
        """
        Функция возвращает смоделированную последовательность в виде <int> элементов
        Returns
        -------
        np.ndarray
            Последовательность
        """
        return self.sequence

    def _set_transition_tensor(self, tensor, rel_tol=1e-6):
        """
        Функция проверки и установки тензора переходов, если такой был задан при инициализации.
        Parameters
        ----------
        tensor : Union[np.ndarray, list], optional
            Тензор переходов
        rel_tol : float
            Максимальное отклонение машинной точности при тестирование на сходимость к 1
        """
        if not isinstance(tensor, np.ndarray) and isinstance(tensor, list):
            tensor = np.array(tensor)
        if tensor.ndim - 1 != self.depth:
            raise ValueError(f'Mismatch at number of dims in tensor and depth, got {tensor.ndim} and {self.depth} '
                             f'corresponding')
        if tensor.size != tensor.shape[0] ** tensor.ndim:
            raise ValueError(f'Tensor dimensions must be equal, got {tensor.shape}')

        index_3d = np.zeros(self.depth + 1, dtype=np.uint32)
        for i in range(self.n ** self.depth):
            _sum = 0.0
            for j in range(self.n):
                _sum += tensor.item(tuple(index_3d))
                index_3d = self._update_3d_index(index_3d)
            if np.abs(1. - _sum) > rel_tol:
                raise ValueError(f'Sum: {_sum} does not coverage not to 1\nSee before {index_3d} index')
        self.transition_tensor = tensor

    def _update_3d_index(self, index):
        """
        Функция обновления индекса тензора произвольного размера в правильном порядке
        Parameters
        ----------
        index : Union[list, np.ndarray]
            Индекс

        Returns
        -------
        Union[list, np.ndarray]
            Обновленный индекс
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
        Функция создания тензора переходов случайным образом
        """
        matrix = np.zeros([len(self.input_p)] * (self.depth + 1), dtype=np.float64)
        index_3d = [0] * (self.depth + 1)
        for _ in tqdm(range(self.n ** self.depth), desc='Transition Tensor Generation'):
            rand_perm = generate_multinomial_rv(self.iter_for_tensor, self._transitions_pvals) / self.iter_for_tensor
            np.random.shuffle(rand_perm)
            for j in range(self.n):
                matrix.itemset(tuple(index_3d), rand_perm[j])
                index_3d = self._update_3d_index(index_3d)
        self.transition_tensor = matrix

    @staticmethod
    def _check_u_in_interval(u, left_bound, right_bound, start=True):
        """
        Проверяет принадлежность равномерно распределенной случайной величины к интервалу в зависимости
        от позиции интервала.
        Для большего понимания см. Ивченко Г.И., Медведев Ю.И. - "Математическая статистика" параграф про моделирование
        полиномиального распределение и цепей Маркова.
        Parameters
        ----------
        u : float
            Равномерно распределенная случайная величина на [0; 1]
        left_bound : float
            Левая граница интервала
        right_bound : float
            Правая граница интервада
        start : bool
            Условие первого интервала

        Returns
        -------
        bool
            Принадлежит ли величина заданному интервалу
        """
        if start:
            return left_bound <= u <= right_bound
        else:
            return left_bound < u <= right_bound

    def _generate_start_transition(self):
        """
        Функция создания первых состояний цепи. Количество начальных состояний зависит от глубины зависимости
        Returns
        -------
        np.ndarray
            Массив начальных состояний
        """
        start_transitions = np.empty(self.depth, dtype=np.uint64)
        intervals = np.append(np.array([0]), np.cumsum(self.input_p))
        for i in range(self.depth):
            u = self._rng.random()
            first_iter = True
            for j in range(intervals.shape[0] - 1):
                if j != 0:
                    first_iter = False
                if self._check_u_in_interval(u, intervals[j], intervals[j + 1], start=first_iter):
                    start_transitions[i] = j
                    break
        return start_transitions

    def _generate_transition(self, prev_transitions):
        """
        Функция моделирования нового состояния цепи в зависимости от предыдущих.
        Parameters
        ----------
        prev_transitions : np.ndarray
            Массив предыдущих состояний

        Returns
        -------
        int
            Новое состояние
        """
        vector = self.transition_tensor
        for i in range(prev_transitions.shape[0]):
            vector = vector[prev_transitions[i]]
        intervals = np.append(np.array([0]), np.cumsum(vector))
        u = self._rng.random()
        first_iter = True
        new_transition = -1
        for j in range(intervals.shape[0] - 1):
            if j != 0:
                first_iter = False
            if self._check_u_in_interval(u, intervals[j], intervals[j + 1], start=first_iter):
                new_transition = j
                break
        if new_transition == -1:
            raise AssertionError(f'While generating new transition came vector {vector}.\nCheck your transition tensor')
        return new_transition

    def generate_chain(self, steps):
        """
        Функция моделирования всей цепи Маркова. Главная функция класса.
        Функция не возвращает значение, а сохраняет ее в self.sequence
        Parameters
        ----------
        steps : int
            Количество моделируемых состояний цепи
        Returns
        -------
        np.ndarray
            Массив состояний цепи
        """
        if isinstance(steps, float):
            print('WARNING: Be careful, type of steps should be <int>. Steps will be transformed into <int>')
        try:
            steps = int(steps)
        except TypeError:
            raise TypeError(f'Expected type of steps is <int>, got {type(steps)}')
        seq = np.empty(steps, dtype=np.uint64)
        seq[0:self.depth] = self._generate_start_transition()
        for i in tqdm(range(self.depth, steps), desc='Chain Generation'):
            seq[i] = self._generate_transition(seq[i - self.depth:i])
        if self.symbols is not None:
            old_seq = seq.copy()
            seq = np.empty_like(old_seq, dtype=object)
            for i in range(old_seq.size):
                seq[i] = self.symbols[old_seq[i]]
        self.sequence = seq
        return seq
