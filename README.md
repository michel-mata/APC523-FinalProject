# Cultural Evolution Numerical Toolkit

A Python library and analysis notebooks for modeling and numerically solving cultural trait dynamics under various social learning frameworks.

## Features

- **Analytical solutions** for equilibrium popularity distributions and persistence times in infinite‑pool models.
- **Modular learning functions** allowing customization of asocial innovation and social transmission rules.
- **Multiple numerical solvers**: direct matrix inversion, linear update, Jacobi, Gauss–Seidel, and Newton–Raphson.
- **Experiment framework** for parameter sweeps, parallel dispatch, and performance/error tracking.
- **Finite‑pool extension** introducing nonlinearity and advanced solver strategies.


## Repository Structure

```bash
.
├── notebooks/
│   ├── analysis.ipynb        # Baseline model experiments and figures
│   └── extensions.ipynb      # Finite‑pool model exploration
├── src/
│   └── cultural_evolution/
│       ├── analytical.py     # Closed‑form solutions
│       ├── system.py         # Transition matrix and learning functions
│       ├── methods.py        # Update routines for solvers
│       ├── solvers.py        # Solver driver and configurations
│       ├── experiments.py    # Parameter grid and experiment runner
│       └── utils.py          # Utility functions (safe div, Jacobian, etc.)
├── tests/
│   ├── test_equilibrium.py   # Unit tests for analytical distributions
│   ├── test_learning.py      # Tests for learning functions
│   └── test_transitions.py   # Tests for transition rates
├── setup.py                  # Package metadata and entry points
├── pyproject.toml            # Build configuration
└── README.md                 # This file
```

## Usage

### Run Experiments

```python
from cultural_evolution.experiments import generate_parameter_grid, run_experiments

# Define parameter grid
params = {
    "N": [100],
    "M": [1,2,3],
    "p_d": [0.1],
    "p_s": [0.1,0.5,0.9]
}
grid = generate_parameter_grid(params)
solver_configs = [{"mode":"popularity","solver":"direct","trace":False}]
summary, solutions = run_experiments(grid, solver_configs, parallel=True, show_progress=True)
```

### Compute Analytical Solutions

```python
from cultural_evolution.analytical import popularity, persistence

# Popularity distribution
f = popularity(N=100, M=3, p_d=0.1, p_s=0.5)
# Persistence times
tau = persistence(N=100, M=3, p_d=0.1, p_s=0.5)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
