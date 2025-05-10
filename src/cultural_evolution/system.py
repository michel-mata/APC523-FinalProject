import numpy as np
from typing import Callable, Tuple
from .utils import *


def get_learning_functions(
    N,
    M,
    p_d: float,
    p_s: float,
    attempt: str = "union",
    sampling: str = "with",
    pool: str = "infinite",
    L: int = None,
    K: int = None,
    f: np.ndarray = None,
) -> Tuple[Callable[[int], float], Callable[[int, int, int], float]]:
    """
    Build asocial (innovation) and social (transmission) learning functions.

    This function constructs two probability functions that govern cultural trait acquisition:
    the asocial innovation rate and the social transmission probability, based on model parameters and modes.

    Parameters
    ----------
    N : int
        Total population size.
    M : int
        Number of role models sampled for social learning.
    p_d : float
        Asocial (innovation) rate.
    p_s : float
        Social transmission probability per attempt.
    attempt : {'union', 'exposures', 'budget', 'public'}, optional
        Mode for social learning attempts:
        - 'union': one shot if encountered.
        - 'exposures': one shot per encounter.
        - 'budget': constrained by learning budget L.
        - 'public': group-level public sampling.
    sampling : {'with', 'without'}, optional
        Sampling of role models with or without replacement.
    pool : {'infinite', 'finite'}, optional
        Innovation pool mode:
        - 'infinite': constant innovation rate.
        - 'finite': density-dependent innovation requiring f and K.
    L : int, optional
        Learning budget (required if attempt is 'budget' or 'public').
    K : int, optional
        Total number of available traits (required if pool is 'finite').
    f : array_like, optional
        Current popularity vector (required if pool is 'finite').

    Returns
    -------
    asocial : callable
        Function asocial(k) -> float giving innovation probability at popularity k.
    social : callable
        Function social(N, M, k) -> float giving social acquisition probability from popularity k.

    Raises
    ------
    ValueError
        If invalid modes are specified or required parameters are missing.

    Examples
    --------
    >>> asoc, soc = get_learning_functions(N=100, M=5, p_d=1e-3, p_s=0.2)
    >>> asoc(0)
    0.001
    >>> soc(100, 5, 10)
    0.198...
    """

    if sampling not in ("with", "without"):
        raise ValueError(f"Unknown sampling mode '{sampling}'")
    if attempt not in ("union", "exposures", "budget", "public"):
        raise ValueError(f"Unknown attempt_mode '{attempt}'")
    if pool not in ("infinite", "finite"):
        raise ValueError(f"Unknown pool mode '{pool}'")

    if pool == "finite":
        if K is None:
            raise ValueError(f"K {K} must be provided for finite pool")
        if f is None:
            f = np.zeros(N, dtype=float)

        def asocial(k: int) -> float:
            total = f.sum()
            rate = p_d * max(0.0, 1.0 - total / K)
            return rate if k == 0 else 0.0

    elif pool == "infinite":

        def asocial(k: int) -> float:
            return p_d if k == 0 else 0.0

    else:
        raise ValueError(f"Unknown pool mode '{pool}'")

    if attempt == "union":

        if sampling == "without":

            def social(N: int, M: int, k: int) -> float:
                p_zero = hypergeom_pmf(0, N, k, M)
                return p_s * (1.0 - p_zero)

        elif sampling == "with":

            def social(N: int, M: int, k: int) -> float:
                p = safe_div(k, N)
                p_zero = binom_pmf(0, M, p)
                return p_s * (1.0 - p_zero)

    elif attempt == "exposures":
        if sampling == "without":

            def social(N: int, M: int, k: int) -> float:
                max_i = min(M, k)
                return sum(
                    hypergeom_pmf(i, N, k, M) * at_least_once(p_s, i)
                    for i in range(1, max_i + 1)
                )

        elif sampling == "with":

            def social(N: int, M: int, k: int) -> float:
                max_i = min(M, k)
                p = safe_div(k, N)
                return sum(
                    binom_pmf(i, M, p) * at_least_once(p_s, i)
                    for i in range(1, max_i + 1)
                )

    elif attempt == "budget":
        if L is None:
            raise ValueError(f"L {L} must be provided for 'budget' mode")
        if f is None:
            f = np.zeros(N, dtype=float)

        ks = np.arange(N) + 1
        C_m = (M / N) * (ks @ f)

        if sampling == "without":

            def social(N: int, M: int, k: int) -> float:
                max_i = min(M, k)
                return sum(
                    hypergeom_pmf(i, N, k, M) * at_least_once(safe_div(p_s * i, C_m), L)
                    for i in range(1, max_i + 1)
                )

        elif sampling == "with":

            def social(N: int, M: int, k: int) -> float:
                max_i = min(M, k)
                p = safe_div(k, N)
                return sum(
                    binom_pmf(i, M, p) * at_least_once(safe_div(p_s * i, C_m), L)
                    for i in range(1, max_i + 1)
                )

    elif attempt == "public":
        if L is None:
            raise ValueError(f"f {f} and L {L} must be provided for 'budget' mode")

        ks = np.arange(len(f)) + 1
        C_n = ks @ f

        def social(N: int, M: int, k: int) -> float:
            return p_s * at_least_once(safe_div(k, C_n), L)

    else:
        raise ValueError(
            f"Unsupported combination attempt='{attempt}', sampling='{sampling}'"
        )

    return asocial, social


def get_transition_functions(
    asocial: Callable[[int], float], social: Callable[[int, int, int], float]
) -> Tuple[Callable[[int, int, int], float], Callable[[int, int, int], float]]:
    """
    Build birth-death transition rate functions.

    This function returns T_plus and T_minus, which compute the rates of
    increasing or decreasing a trait's popularity in the population.

    Parameters
    ----------
    asocial : callable
        Function asocial(k) -> float for innovation probability.
    social : callable
        Function social(N, M, k) -> float for social acquisition probability.

    Returns
    -------
    T_plus : callable
        Function T_plus(N, M, k) -> float for transition k->k+1 probability.
    T_minus : callable
        Function T_minus(N, M, k) -> float for transition k->k-1 probability.

    Examples
    --------
    >>> T_plus, T_minus = get_transition_functions(asoc, soc)
    >>> T_plus(100, 5, 10)
    0.017...
    """

    def loss(N_: int, k_: int) -> float:
        return safe_div(k_, N_)

    def T_plus(N: int, M: int, k: int) -> float:
        p_learning = asocial(k) + social(N - 1, M, k)
        p_loss = loss(N, k)
        return p_learning * (1 - p_loss)

    def T_minus(N: int, M: int, k: int) -> float:
        p_learning = social(N - 1, M, k - 1)
        p_loss = loss(N, k)
        return (1 - p_learning) * p_loss

    return T_plus, T_minus


def get_transition_matrix(
    N: int,
    M: int,
    p_d: float,
    p_s: float,
    attempt: str = "union",
    sampling: str = "with",
    pool: str = "infinite",
    K: int = None,
    L: int = None,
    f: np.ndarray = None,
    return_system: bool = True,
) -> np.ndarray:
    """
    Construct the transition matrix (and source vector) for the birth-death process.

    Each step moves traits between popularity levels based on birth and death
    rates, assembled into a transition probability matrix P.

    Parameters
    ----------
    N : int
        Number of popularity levels.
    M : int
        Number of role models sampled.
    p_d : float
        Innovation rate.
    p_s : float
        Social learning probability.
    attempt : {'union', 'exposures', 'budget', 'public'}, optional
        Social learning attempt mode.
    sampling : {'with', 'without'}, optional
        Role model sampling replacement.
    pool : {'infinite', 'finite'}, optional
        Innovation pool mode.
    K : int, optional
        Number of available traits (required for finite pool).
    L : int, optional
        Learning budget (required if attempt is 'budget' or 'public').
    f : array_like, optional
        Current popularity state vector for finite pool calculations.
    return_system : bool, default True
        If True, return tuple (P, b); otherwise return P only.

    Returns
    -------
    P : ndarray, shape (N, N)
        Transition probability matrix among popularity levels.
    b : ndarray, shape (N,)
        Source vector for innovations (only if return_system is True).

    Examples
    --------
    >>> P, b = get_transition_matrix(100, 5, 1e-3, 0.2)
    >>> P.shape
    (100, 100)
    """

    p_asocial, p_social = get_learning_functions(
        N, M, p_d, p_s, attempt=attempt, sampling=sampling, pool=pool, L=L, K=K, f=f
    )
    T_plus, T_minus = get_transition_functions(p_asocial, p_social)

    P = np.zeros((N, N), dtype=float)
    for k in range(N):
        increase = T_plus(N, M, k + 1)
        decrease = T_minus(N, M, k + 1)
        stay = 1 - (increase + decrease)

        if k > 0:
            P[k, k - 1] = decrease

        if k < N - 1:
            P[k, k + 1] = increase

        P[k, k] = stay

    if return_system:
        b = np.zeros(N, dtype=float)
        b[0] = p_asocial(0)
        return P, b
    else:
        return P
