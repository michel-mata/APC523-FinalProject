import numpy as np
from .utils import *
from .system import get_transition_matrix
from .methods import (
    linear_update,
    jacobi_update,
    gauss_seidel_update,
    nonlinear_update,
    newton_update,
)


def direct_solver(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear system A x = b directly.

    This function uses NumPy's `linalg.solve` to compute the solution
    of the linear system

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Coefficient matrix.
    b : ndarray, shape (N,)
        Right-hand side vector.

    Returns
    -------
    x : ndarray, shape (N,)
        Solution vector.

    Examples
    --------
    >>> import numpy as np
    >>> x = direct_solver(np.eye(3), np.array([1, 2, 3]))
    >>> np.allclose(x, [1, 2, 3])
    True
    """

    return np.linalg.solve(A, b)


def fixed_point_solver(
    update_func,
    x: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = int(1e4),
    trace: bool = True,
) -> np.ndarray:
    """
    Generic fixed-point iteration solver.

    This function repeatedly applies an update function until the solution
    converges within the specified tolerance or until the maximum number
    of iterations is reached.

    Parameters
    ----------
    update_func : callable
        Function mapping the current guess to the next guess.
    x : ndarray, shape (N,)
        Initial guess vector.
    tol : float, optional
        Convergence tolerance based on the infinity norm. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations. Default is 1e4.
    trace : bool, optional
        If True, return the residual history alongside the solution. Default is True.

    Returns
    -------
    x : ndarray, shape (N,)
        Converged solution vector.
    residuals : list of float
        Infinity-norm residuals for each iteration (if `trace` is True).

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda v: 0.5 * (v + 2/v)
    >>> x, res = fixed_point_solver(f, np.array([1.0]), tol=1e-6, max_iter=10, trace=True)
    """

    no_iter = max_iter == 0
    residuals = []
    i = 0
    res = np.inf
    x = x.copy()

    while res > tol and (no_iter or i < max_iter):
        i += 1

        x_new = update_func(x)
        res = np.linalg.norm(x_new - x, np.inf)
        x = x_new

        residuals.append(res)

    if trace:
        return x, residuals
    else:
        return x


def solve_model(
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
    mode: str = "popularity",
    solver: str = "linear",
    tol: float = 1e-8,
    max_iter: int = int(1e4),
    trace: bool = True,
) -> np.ndarray:
    """
    General solver for equilibrium popularity distribution or persistence times.

    This function assembles the transition system, chooses a numerical solver,
    and computes either the equilibrium popularity distribution (mode='popularity')
    or expected extinction times (mode='persistence').

    Parameters
    ----------
    N : int
        Number of popularity levels (population size).
    M : int
        Number of role models sampled.
    p_d : float
        Asocial (innovation) rate.
    p_s : float
        Social learning probability per attempt.
    attempt : {'union', 'exposures', 'budget', 'public'}, optional
        Social learning attempt mode. Default is 'union'.
    sampling : {'with', 'without'}, optional
        Role model sampling replacement mode. Default is 'with'.
    pool : {'infinite', 'finite'}, optional
        Innovation pool mode. Default is 'infinite'.
    K : int, optional
        Number of available traits (required if pool is 'finite').
    L : int, optional
        Learning budget (required if attempt is 'budget' or 'public').
    f : ndarray, optional
        Current popularity state vector for finite pool.
    mode : {'popularity', 'persistence'}, optional
        Type of solution to compute. Default is 'popularity'.
    solver : {'direct', 'linear', 'jacobi', 'gauss_seidel', 'nonlinear', 'newton'}, optional
        Numerical solver to use. Default is 'linear'.
    tol : float, optional
        Convergence tolerance for iterative methods. Default is 1e-8.
    max_iter : int, optional
        Maximum iterations for iterative solvers. Default is 1e4.
    trace : bool, optional
        If True, return residual history along with the solution. Default is True.

    Returns
    -------
    x : ndarray, shape (N,)
        Solution vector: equilibrium popularity distribution or persistence times.
    residuals : list of float
        Residual history (for iterative solvers) or [nan] for direct solver.

    Raises
    ------
    ValueError
        If `mode` or `solver` is not recognized.

    Examples
    --------
    >>> f, res = solve_model(100, 5, 1e-3, 0.2, solver='direct')
    >>> len(f), len(res)
    (100, 1)
    """
    if (pool == "finite" or attempt == "budget") and (
        solver not in ("nonlinear", "newton")
    ):
        import warnings

        warnings.warn(
            "Finite pool or budget attempt mode requires a non-linear solver; "
            f"overriding solver '{solver}' to 'newton'."
        )
        solver = "newton"

    P, b = get_transition_matrix(
        N,
        M,
        p_d,
        p_s,
        attempt=attempt,
        sampling=sampling,
        pool=pool,
        L=L,
        K=K,
        f=f,
        return_system=True,
    )

    if mode == "popularity":
        Q = P.T
    elif mode == "persistence":
        Q = P
        b = np.ones(N, dtype=float)
    else:
        raise ValueError(
            f"Unknown mode '{mode}', expected 'popularity' or 'persistence'"
        )
    A = np.eye(N) - Q

    if solver == "direct":
        return direct_solver(A, b), [np.nan]
    elif solver == "linear":
        update_func = linear_update(Q, b)
    elif solver == "jacobi":
        update_func = jacobi_update(A, b)
    elif solver == "gauss_seidel":
        update_func = gauss_seidel_update(A, b)
    elif solver == "nonlinear":
        update_func = nonlinear_update(
            N=N,
            M=M,
            p_d=p_d,
            p_s=p_s,
            attempt=attempt,
            sampling=sampling,
            pool=pool,
            K=K,
            L=L,
        )
    elif solver == "newton":
        update_func = newton_update(
            N=N,
            M=M,
            p_d=p_d,
            p_s=p_s,
            attempt=attempt,
            sampling=sampling,
            pool=pool,
            K=K,
            L=L,
        )
    else:
        raise ValueError(f"Unknown solver '{solver}'")

    return fixed_point_solver(
        update_func, b.copy(), tol=tol, max_iter=max_iter, trace=trace
    )
