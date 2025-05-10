import numpy as np
from math import prod
from .utils import *
from .system import get_learning_functions, get_transition_functions


def popularity(
    N: int,
    M: int,
    p_d: float,
    p_s: float,
    attempt: str = "union",
    sampling: str = "with",
    pool: str = "infinite",
    K: int = None,
) -> np.ndarray:
    """
    Compute equilibrium popularity distribution of cultural traits.

    This function returns the unnormalized popularity counts f_k for trait frequencies
    k = 1...N in a finite population under combined innovation and social learning.

    Parameters
    ----------
    N : int
        Maximum popularity level (population size).
    M : int
        Number of role models sampled for social learning.
    p_d : float
        Asocial innovation rate.
    p_s : float
        Social learning probability per attempt.
    attempt : {'union', 'exposures', 'budget', 'public'}, optional
        Mode for social learning attempts.
    sampling : {'with', 'without'}, optional
        Role model sampling replacement mode.

    Returns
    -------
    f : ndarray, shape (N,)
        Unnormalized popularity distribution for k = 1...N.

    Examples
    --------
    >>> f = popularity(N=100, M=5, p_d=1e-3, p_s=0.2)
    >>> f.shape
    (100,)
    """
    asocial, social = get_learning_functions(N, M, p_d, p_s, attempt, sampling)
    T_plus, T_minus = get_transition_functions(asocial, social)
    ratios = [safe_div(T_plus(N, M, j), T_minus(N, M, j + 1)) for j in range(N)]
    f = np.array([prod(ratios[: k + 1]) for k in range(N)])
    if pool == "infinite":
        return f
    elif pool == "finite":
        if K is None:
            raise ValueError(f"K {K} must be provided for finite pool")
        C = f.sum()
        p_d_eff = p_d / (1 + C / K)
        asocial, social = get_learning_functions(N, M, p_d_eff, p_s, attempt, sampling)
        T_plus, T_minus = get_transition_functions(asocial, social)
        ratios = [safe_div(T_plus(N, M, j), T_minus(N, M, j + 1)) for j in range(N)]
        f = np.array([prod(ratios[: k + 1]) for k in range(N)])
        return f


def persistence(
    N: int,
    M: int,
    p_d: float,
    p_s: float,
    attempt: str = "union",
    sampling: str = "with",
) -> np.ndarray:
    """
    Compute expected extinction times for cultural traits.

    This function calculates the expected time to extinction τ_k for traits starting
    at popularity k = 1...N, using prefix-product and prefix-sum optimizations.

    Parameters
    ----------
    N : int
        Maximum popularity level (population size).
    M : int
        Number of role models sampled for social learning.
    p_d : float
        Asocial innovation rate.
    p_s : float
        Social learning probability per attempt.
    attempt : {'union', 'exposures', 'budget', 'public'}, optional
        Mode for social learning attempts.
    sampling : {'with', 'without'}, optional
        Role model sampling replacement mode.

    Returns
    -------
    tau : ndarray, shape (N,)
        Expected extinction times for initial popularity levels 1...N.

    Examples
    --------
    >>> tau = persistence(N=100, M=5, p_d=1e-3, p_s=0.2)
    >>> tau[0]
    50.3...
    """
    p_asocial, p_social = get_learning_functions(N, M, p_d, p_s, attempt, sampling)
    T_plus, T_minus = get_transition_functions(p_asocial, p_social)

    ratios = [safe_div(T_plus(N, M, j), T_minus(N, M, j + 1)) for j in range(1, N + 1)]
    factors = [safe_div(1, T_minus(N, M, i)) for i in range(1, N + 1)]

    tau = [
        factors[i] * sum(prod(ratios[i : k - 1]) for k in range(i, N)) for i in range(N)
    ]

    return np.cumsum(np.array(tau))
