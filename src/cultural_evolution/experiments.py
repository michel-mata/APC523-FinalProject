import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd


from .solvers import solve_model
from .analytical import popularity, persistence


# Module-level worker function for parallel execution
def _run_one_task(params, config):
    """
    Worker task for run_experiments: takes params dict and solver config dict,
    runs the model, and returns (rec, sol).
    """
    import time
    import numpy as np
    from .solvers import solve_model
    from .analytical import popularity, persistence
    from .experiments import compute_error_metrics

    start = time.time()
    sol, residuals = solve_model(
        **params,
        mode=config["mode"],
        solver=config["solver"],
        trace=config.get("trace", False)
    )
    elapsed = time.time() - start
    rec = dict(params)
    rec.update(
        {
            "mode": config["mode"],
            "solver": config["solver"],
            "time": elapsed,
            "iterations": len(residuals) if isinstance(residuals, list) else np.nan,
        }
    )
    # Compute analytical reference only for infinite pool
    if config["mode"] == "popularity":
        ref = popularity(
            N=params["N"],
            M=params["M"],
            p_d=params["p_d"],
            p_s=params["p_s"],
            attempt=params.get("attempt", "union"),
            sampling=params.get("sampling", "with"),
        )
    else:
        ref = persistence(
            N=params["N"],
            M=params["M"],
            p_d=params["p_d"],
            p_s=params["p_s"],
            attempt=params.get("attempt", "union"),
            sampling=params.get("sampling", "with"),
        )
    err_L2, err_inf = compute_error_metrics(sol, ref)
    rec["error_L2"] = err_L2
    rec["error_inf"] = err_inf
    return rec, sol


def generate_parameter_grid(param_dict):
    """
    Generate a list of parameter dictionaries for all combinations of input lists.

    Parameters
    ----------
    param_dict : dict
        Mapping of parameter names to lists of values.

    Returns
    -------
    grid : list of dict
        List of all combinations of parameters.

    Examples
    --------
    >>> grid = generate_parameter_grid({'N': [50, 100], 'M': [5, 10]})
    >>> len(grid)
    4
    """
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    return grid


def normalize(f):
    """
    Normalize an array to sum to 1.

    Parameters
    ----------
    f : ndarray
        Input array.

    Returns
    -------
    p : ndarray
        Normalized array summing to 1.
    """
    total = np.sum(f)
    return f / total if total != 0 else f


def shannon_entropy(f):
    """
    Compute Shannon entropy of a discrete distribution.

    Parameters
    ----------
    f : ndarray
        Input counts or weights.

    Returns
    -------
    H : float
        Shannon entropy.
    """
    p = normalize(f)
    return -np.sum(p * np.log(p + 1e-12))


def simpson_index(f):
    """
    Compute Simpson diversity index of a discrete distribution.

    Parameters
    ----------
    f : ndarray
        Input counts or weights.

    Returns
    -------
    D : float
        Simpson index = 1 - sum(p^2).
    """
    p = normalize(f)
    return 1.0 - np.sum(p**2)


def compute_error_metrics(numerical, analytical):
    """
    Compute L2 and infinity-norm errors between numerical and analytical arrays.

    Parameters
    ----------
    numerical : ndarray
        Numerical solution vector.
    analytical : ndarray
        Reference analytical solution.

    Returns
    -------
    error_L2 : float
        L2-norm of the difference.
    error_inf : float
        Infinity-norm of the difference.
    """
    diff = numerical - analytical
    error_L2 = np.linalg.norm(diff)
    error_inf = np.max(np.abs(diff))
    return error_L2, error_inf


def run_experiments(
    param_grid, solver_configs, parallel=True, max_workers=None, show_progress=False
):
    """
    Run a suite of experiments over parameter grid and solver configurations.

    Parameters
    ----------
    param_grid : list of dict
        List of parameter sets for solve_model.
    solver_configs : list of dict
        Each dict must contain 'mode' and 'solver', and optional 'trace'.
    parallel : bool, optional
        If True, use ProcessPoolExecutor. Default is True.
    max_workers : int or None, optional
        Number of worker processes. If None, defaults to os.cpu_count().

    Returns
    -------
    summary : pandas.DataFrame
        DataFrame with one row per run, including parameters, mode, solver,
        time, iterations, error_L2, error_inf.
    solutions : pandas.DataFrame
        DataFrame with raw solution vectors and total_traits for each run.
    """
    summary_records = []
    solution_records = []

    tasks = [(p, c) for p in param_grid for c in solver_configs]

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_run_one_task, p, c) for p, c in tasks]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator, total=len(futures), desc="Running experiments"
                )
            for fut in iterator:
                rec, sol = fut.result()
                summary_records.append(rec)
                sol_rec = rec.copy()
                sol_rec["solution"] = sol
                sol_rec["total_traits"] = float(np.sum(sol))
                solution_records.append(sol_rec)
    else:
        for p, c in tasks:
            rec, sol = _run_one_task(p, c)
            summary_records.append(rec)
            sol_rec = rec.copy()
            sol_rec["solution"] = sol
            sol_rec["total_traits"] = float(np.sum(sol))
            solution_records.append(sol_rec)

    summary_df = pd.DataFrame(summary_records)
    solutions_df = pd.DataFrame(solution_records)
    return summary_df, solutions_df


class Experiment:
    """
    High-level experiment wrapper for a fixed parameter set.
    """

    def __init__(self, **params):
        """
        Store model parameters.

        Parameters
        ----------
        params : keyword args
            Parameters to pass to solve_model and analytical functions.
        """
        self.params = params

    def run(self, mode, solver, trace=False):
        """
        Run solve_model for the given mode and solver.

        Returns result, residual history, and elapsed time.
        """
        start = time.time()
        sol, residuals = solve_model(
            **self.params, mode=mode, solver=solver, trace=trace
        )
        elapsed = time.time() - start
        return sol, residuals, elapsed

    def error(self, mode, sol):
        """
        Compute error metrics against analytical solution.

        Parameters
        ----------
        mode : str
            'popularity' or 'persistence'.
        sol : ndarray
            Numerical solution vector.

        Returns
        -------
        error_L2, error_inf : float
        """
        if mode == "popularity":
            ref = popularity(**self.params)
        else:
            ref = persistence(**self.params)
        return compute_error_metrics(sol, ref)

    def run_all(self, solver_configs):
        """
        Run all solver configurations and collect results.

        Returns
        -------
        pandas.DataFrame
        """
        records = []
        for config in solver_configs:
            sol, residuals, elapsed = self.run(
                config["mode"], config["solver"], trace=config.get("trace", False)
            )
            err_L2, err_inf = self.error(config["mode"], sol)
            rec = dict(self.params)
            rec.update(
                {
                    "mode": config["mode"],
                    "solver": config["solver"],
                    "time": elapsed,
                    "iterations": (
                        len(residuals) if isinstance(residuals, list) else np.nan
                    ),
                    "error_L2": err_L2,
                    "error_inf": err_inf,
                }
            )
            records.append(rec)
        return pd.DataFrame(records)
