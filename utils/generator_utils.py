def random_stochastic_vector(n: int, left_bound=0, right_bound=999, seed=None):
    """
    Функция создания стахостического вектора из полиномиального распределения с количество элементов <n>.
    Parameters
    ----------
    n : int
        Количество элементов для полиномиальной схемы
    left_bound : int
        Левая граница
    right_bound : int
        Правая граница
    seed : optional
        Зерно ГПСЧ

    Returns
    -------
    np.ndarray
        Стахостический вектор
    """
    rng = np.random.default_rng(seed=seed)
    random_int_vec = np.zeros(n)
    for i in range(n):
        random_int_vec[i] = rng.integers(left_bound, right_bound)
    _sum = random_int_vec.sum()
    return random_int_vec / _sum
