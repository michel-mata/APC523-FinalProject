import numpy as np
from typing import Callable
from .utils import *
from .system import (
    get_transition_matrix,
    get_learning_functions,
    get_transition_functions,
)


def linear_update(A: np.ndarray, b: np.ndarray):
    """
    Return a linear update function x -> A @ x + b.

    This function constructs a fixed-point linear operator for solving iterative systems of the form x = A x + b.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Coefficient matrix for the linear part.
    b : ndarray, shape (N,)
        Constant vector for the affine term.

    Returns
    -------
    update : callable
        Function update(x: ndarray) -> ndarray computing A @ x + b.

    Examples
    --------
    >>> import numpy as np
    >>> update = linear_update(np.eye(2), np.array([1,2]))
    >>> update(np.zeros(2))
    array([1, 2])
    """

    def update(x: np.ndarray) -> np.ndarray:
        return A @ x + b

    return update


def jacobi_update(A: np.ndarray, b: np.ndarray):
    """
    Return a Jacobi update function for solving A x = b.

    This function constructs the Jacobi method update: x_new = (b - R x) / D,
    where A = D + R with D diagonal.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Coefficient matrix.
    b : ndarray, shape (N,)
        Right-hand side vector.

    Returns
    -------
    update : callable
        Function update(x: ndarray) -> ndarray performing one Jacobi iteration.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4,1],[1,3]])
    >>> b = np.array([1,2])
    >>> update = jacobi_update(A, b)
    >>> x0 = np.zeros(2)
    >>> update(x0)
    array([0.25, 0.66666667])
    """
    M = np.diag(A)
    N = A - np.diagflat(M)

    def update(x: np.ndarray) -> np.ndarray:
        return (b - N @ x) / M

    return update


def gauss_seidel_update(A: np.ndarray, b: np.ndarray):
    """
    Return a Gauss-Seidel update function for solving A x = b.

    This function constructs the Gauss-Seidel iteration which updates each component sequentially.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Coefficient matrix.
    b : ndarray, shape (N,)
        Right-hand side vector.

    Returns
    -------
    update : callable
        Function update(x: ndarray) -> ndarray performing one Gauss-Seidel iteration.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4,1],[1,3]])
    >>> b = np.array([1,2])
    >>> update = gauss_seidel_update(A, b)
    >>> x0 = np.zeros(2)
    >>> update(x0)
    array([0.25, 0.58333333])
    """
    N = A.shape[0]

    def update(x: np.ndarray) -> np.ndarray:
        x_new = x.copy()
        for i in range(N):
            sum1 = A[i, :i] @ x_new[:i]
            sum2 = A[i, i + 1 :] @ x[i + 1 :]
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        return x_new

    return update


def nonlinear_update(
    N: int,
    M: int,
    p_d: float,
    p_s: float,
    attempt: str,
    sampling: str,
    pool: str,
    K: int,
    L: int,
):
    """
    Return a nonlinear fixed-point update f -> P(f).T f + b(f).

    This function constructs an update operator for the nonlinear system f = P(f).T f + b(f),
    where P and b depend on current f via model parameters.

    Parameters
    ----------
    N : int
        Number of popularity levels.
    M : int
        Number of role models sampled.
    p_d : float
        Asocial innovation rate.
    p_s : float
        Social learning probability per attempt.
    attempt : {'union', 'exposures', 'budget', 'public'}
        Social learning attempt mode.
    sampling : {'with', 'without'}
        Role model sampling replacement mode.
    pool : {'infinite', 'finite'}
        Innovation pool mode.
    K : int or None
        Total number of available traits for finite pool (required if pool='finite').
    L : int or None
        Learning budget for 'budget' or 'public' modes.

    Returns
    -------
    update : callable
        Function update(f: ndarray) -> ndarray computing one nonlinear fixed-point iteration.

    Examples
    --------
    >>> import numpy as np
    >>> update = nonlinear_update(100, 5, 1e-3, 0.2, 'union', 'with', 'infinite', None, None)
    >>> f0 = np.ones(100) / 100
    >>> f1 = update(f0)
    >>> f1.shape
    (100,)
    """

    def update(f: np.ndarray) -> np.ndarray:
        P, b = get_transition_matrix(
            N,
            M,
            p_d,
            p_s,
            attempt=attempt,
            sampling=sampling,
            pool=pool,
            K=K,
            L=L,
            f=f,
            return_system=True,
        )
        return P.T @ f + b

    return update


def newton_update(
    N: int,
    M: int,
    p_d: float,
    p_s: float,
    attempt: str,
    sampling: str,
    pool: str,
    K: int,
    L: int,
):
    """
    Return a Newton–Raphson update for solving F(f)=0.

    This function constructs one Newton correction step for the nonlinear
    fixed-point function F(f) = P(f).T f + b(f) - f.

    Parameters
    ----------
    N, M, p_d, p_s, attempt, sampling, pool, K, L : as in nonlinear_update

    Returns
    -------
    update : callable
        Function update(f: ndarray) -> ndarray performing one Newton–Raphson iteration.

    Examples
    --------
    >>> import numpy as np
    >>> update = newton_update(100, 5, 1e-3, 0.2, 'union', 'with', 'infinite', None, None)
    >>> f0 = np.ones(100) / 100
    >>> f1 = update(f0)
    >>> f1.shape
    (100,)
    """

    def _update(f: np.ndarray) -> np.ndarray:
        P, b = get_transition_matrix(
            N,
            M,
            p_d,
            p_s,
            attempt=attempt,
            sampling=sampling,
            pool=pool,
            K=K,
            L=L,
            f=f,
            return_system=True,
        )
        return P.T @ f + b

    def update(f: np.ndarray) -> np.ndarray:
        F0 = _update(f) - f
        F_func = lambda x: _update(x) - x
        J = numeric_jacobian(F_func, f)
        delta = np.linalg.solve(J, -F0)
        return f + delta

    return update


def power_iteration_update(A: np.ndarray):
    """
    Return a power iteration update function x -> normalized A @ x.

    This function constructs one step of the power iteration to compute the
    leading eigenvector of A, normalizing by the L1 norm.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Matrix whose dominant eigenvector is sought.

    Returns
    -------
    update : callable
        Function update(x: ndarray) -> ndarray performing one power iteration.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2,0],[0,1]])
    >>> update = power_iteration_update(A)
    >>> x0 = np.array([1.0,1.0])
    >>> update(x0)
    array([0.666..., 0.333...])
    """

    def update(x: np.ndarray) -> np.ndarray:
        y = A @ x
        return y / np.linalg.norm(y, ord=1)

    return update


def relaxed_update(update_func: Callable[[np.ndarray], np.ndarray], omega: float):
    """
    Return a relaxed fixed-point update F_relaxed(x) = omega*F(x) + (1-omega)*x.

    This function wraps an existing update function with relaxation parameter omega.

    Parameters
    ----------
    update_func : callable
        Original fixed-point update function.
    omega : float
        Relaxation coefficient between 0 and 1.

    Returns
    -------
    update : callable
        Function update(x: ndarray) -> ndarray performing the relaxed update.

    Examples
    --------
    >>> import numpy as np
    >>> base = lambda x: x/2
    >>> update = relaxed_update(base, 0.5)
    >>> update(np.array([2.0]))
    array([1.0])
    """

    def update(x: np.ndarray) -> np.ndarray:
        return omega * update_func(x) + (1.0 - omega) * x

    return update
