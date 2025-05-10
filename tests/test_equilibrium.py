import numpy as np
import pytest

from cultural_evolution.popularity import equilibrium


@pytest.mark.parametrize(
    "N,M,p_d,p_s",
    [
        (1, 5, 0.1, 0.3),
        (2, 4, 0.05, 0.2),
        (3, 2, 0.2, 0.5),
        (5, 5, 0.01, 0.1),
    ],
)
def test_equilibrium_basic(N, M, p_d, p_s):
    """
    For various small N, M, p_s, p_d, equilibrium should:
    - return a vector of length N
    - contain only non-negative values
    """
    f = equilibrium(N, M, p_d, p_s)
    assert isinstance(f, np.ndarray)
    assert f.shape == (N,)
    assert np.all(f >= 0)


def test_equilibrium_N1_trivial():
    """
    N=1: there is only one popularity state; it must have f[0] > 0.
    """
    f = equilibrium(1, 10, 0.05, 0.5)
    assert len(f) == 1
    assert f[0] > 0


def test_equilibrium_monotonicity():
    """
    For p_s=0 and p_d small, the mass should concentrate at k=1:
    so f[1:] should be <= f[0].
    """
    N, M, p_d, p_s = 5, 3, 0.1, 0.0
    f = equilibrium(N, M, p_d, p_s)
    assert f[0] >= max(f[1:])
