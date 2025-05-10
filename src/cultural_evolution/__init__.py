import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

# Dynamically import all modules in the current package
package_dir = Path(__file__).resolve().parent
for module_info in pkgutil.iter_modules([str(package_dir)]):
    module = importlib.import_module(f".{module_info.name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        # Add only functions to the global namespace
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            globals()[name] = obj

# Optionally, define __all__ for better control
__all__ = [name for name in globals() if not name.startswith("_")]


# ALTERNATIVE: Explicitly import functions
# from .popularity import equilibrium
# from .persistence import persistence_times, persistence_times_matrix
# from .system import (
#     make_learning_functions,
#     make_transition_functions,
#     build_transition_matrix,
#     safe_div,
# )

# __all__ = [
#     "equilibrium",
#     "persistence_times",
#     "persistence_times_matrix",
#     "make_learning_functions",
#     "make_transition_functions",
#     "build_transition_matrix",
#     "safe_div",
# ]
