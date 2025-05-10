import numpy as np
from scipy.special import comb


def safe_div(num, den):
    """
    Safely divide numerator by denominator, returning zero where denominator is zero.

    Parameters
    ----------
    num : scalar or array_like
        Numerator.
    den : scalar or array_like
        Denominator.

    Returns
    -------
    result : ndarray
        Elementwise division result, with zeros where den == 0.

    Examples
    --------
    >>> safe_div([1, 2], [1, 0])
    array([1., 0.])
    """
    num = np.asarray(num)
    den = np.asarray(den)

    if den.ndim == 0:
        if den == 0:
            return np.zeros_like(num, dtype=float)
        return num / den

    result = np.zeros_like(num, dtype=float)
    nonzero = den != 0
    result[nonzero] = num[nonzero] / den[nonzero]

    return result


def at_least_once(p, n):
    """
    Probability of at least one success in n independent Bernoulli trials.

    Parameters
    ----------
    p : float or array_like
        Success probability for each trial.
    n : int
        Number of independent trials.

    Returns
    -------
    prob : float or ndarray
        Probability of at least one success.

    Examples
    --------
    >>> at_least_once(0.1, 3)
    0.271
    """
    return 1.0 - (1.0 - p) ** n


def binom_pmf(k, n, p):
    """
    Binomial probability mass function.

    Parameters
    ----------
    k : int or array_like
        Number of successes.
    n : int
        Number of trials.
    p : float
        Success probability in each trial.

    Returns
    -------
    pmf : ndarray
        Binomial PMF evaluated at k, with zeros for values outside [0, n].

    Examples
    --------
    >>> binom_pmf([0, 1, 2], 2, 0.5)
    array([0.25, 0.5, 0.25])
    """
    k = np.asarray(k)
    pmf = np.zeros_like(k, dtype=float)
    valid = (k >= 0) & (k <= n)

    pmf[valid] = comb(n, k[valid]) * (p ** k[valid]) * ((1.0 - p) ** (n - k[valid]))

    return pmf


def hypergeom_pmf(k, N, K, n):
    """
    Hypergeometric probability mass function.

    Parameters
    ----------
    k : int or array_like
        Number of observed successes.
    N : int
        Population size.
    K : int
        Total number of successes in population.
    n : int
        Number of draws.

    Returns
    -------
    pmf : ndarray
        Hypergeometric PMF at k, zeros outside support [max(0, n + K - N), min(K, n)].

    Examples
    --------
    >>> hypergeom_pmf([0, 1], 10, 5, 2)
    array([0.1666..., 0.5555...])
    """
    k = np.asarray(k)
    pmf = np.zeros_like(k, dtype=float)

    low = max(0, n + K - N)
    high = min(K, n)

    valid = (k >= low) & (k <= high)

    pmf[valid] = safe_div(comb(K, k[valid]) * comb(N - K, n - k[valid]), comb(N, n))

    return pmf


def trajectory(A: np.ndarray, b: np.ndarray, steps: int = 50) -> np.ndarray:
    """
    Simulate linear dynamical trajectory x_{t+1} = A^T x_t + b.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Transition matrix.
    b : ndarray, shape (N,)
        Constant input vector.
    steps : int, optional
        Number of time steps to simulate. Default is 50.

    Returns
    -------
    traj : ndarray, shape (steps+1, N)
        Trajectory of states from t=0 to t=steps, with initial state b.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.eye(2)
    >>> b = np.ones(2)
    >>> traj = trajectory(A, b, steps=2)
    >>> traj.shape
    (3, 2)
    """
    N = b.size
    traj = np.zeros((steps + 1, N))
    traj[0] = b.copy()
    for t in range(steps):
        traj[t + 1] = A.T @ traj[t] + b
    return traj


def numeric_jacobian(F_func: callable, f: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute numerical Jacobian of a vector function via forward finite differences.

    Parameters
    ----------
    F_func : callable
        Function mapping f (ndarray of shape (N,)) to array of shape (N,).
    f : ndarray, shape (N,)
        Point at which to compute Jacobian.
    eps : float, optional
        Finite difference step size. Default is 1e-6.

    Returns
    -------
    J : ndarray, shape (N, N)
        Approximate Jacobian matrix dF/df at f.

    Examples
    --------
    >>> import numpy as np
    >>> F = lambda x: np.array([2*x[0], 3*x[1]])
    >>> numeric_jacobian(F, np.array([1., 2.]))
    array([[2., 0.],
           [0., 3.]])
    """
    N = f.size
    J = np.zeros((N, N), dtype=float)
    F0 = F_func(f)

    for j in range(N):
        df = np.zeros_like(f)
        df[j] = eps
        Fj = F_func(f + df)
        J[:, j] = (Fj - F0) / eps

    return J
