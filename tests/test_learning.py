from math import comb

import pytest

from cultural_evolution.popularity import make_learning_functions


def test_asocial_learning():
    """
    Asocial (innovation) learning should return p_d at k=0, zero otherwise.
    """
    p_d = 0.1
    p_s = 0.5
    p_asocial, p_social = make_learning_functions(p_d, p_s)
    # Innovation only at k=0
    assert pytest.approx(p_asocial(0), rel=1e-8) == p_d
    for k in [1, 2, 5, 10]:
        assert pytest.approx(p_asocial(k), abs=0.0) == 0.0


@pytest.mark.parametrize("sampling", ["with", "without"])
def test_social_single_shot(sampling):
    """
    Single attempt per group (L=1), single group (G=1):
    p_social = p_s * p_encounter.
    """
    p_d = 0.0
    p_s = 0.4
    L = 1
    G = 1
    # Build the functions
    p_asocial, p_social = make_learning_functions(p_d, p_s, sampling, L, G)
    # Test on small example
    N, M, k = 6, 3, 2
    # encounter from one draw of M from N-1
    if sampling == "with":
        pe = 1 - (1 - k / (N - 1)) ** M
    else:
        # without replacement: probability at least one hit
        pe = 1 - comb((N - 1) - k, M) / comb(N - 1, M)
    expected = p_s * pe
    assert pytest.approx(p_social(N, M, k), rel=1e-8) == expected


def test_social_multi_attempts_same_group():
    """
    Multiple attempts (L>1) on same group (G=1):
    p_social = p_encounter * (1 - (1-p_s)**L)
    """
    p_d = 0.0
    p_s = 0.3
    sampling = "with"
    L = 4
    G = 1
    _, p_social = make_learning_functions(p_d, p_s, sampling, L, G)
    N, M, k = 10, 5, 3
    pe = 1 - (1 - k / (N - 1)) ** M
    expected = pe * (1 - (1 - p_s) ** L)
    assert pytest.approx(p_social(N, M, k), rel=1e-8) == expected


def test_social_multi_groups_single_attempt():
    """
    Single attempt per group (L=1) across multiple groups (G>1):
    p_social = 1 - (1 - p_s * p_encounter)**G
    """
    p_d = 0.0
    p_s = 0.2
    sampling = "with"
    L = 1
    G = 5
    _, p_social = make_learning_functions(p_d, p_s, sampling, L, G)
    N, M, k = 8, 4, 2
    pe = 1 - (1 - k / (N - 1)) ** M
    expected = 1 - (1 - p_s * pe) ** G
    assert pytest.approx(p_social(N, M, k), rel=1e-8) == expected


def test_social_full_generality():
    """
    Combined multi-attempt and multi-group:
    p_social = 1 - (1 - pe*(1-(1-p_s)**L))**G
    """
    p_d = 0.0
    p_s = 0.25
    sampling = "without"
    L = 3
    G = 4
    _, p_social = make_learning_functions(p_d, p_s, sampling, L, G)
    N, M, k = 9, 4, 3
    # without replacement
    pe = 1 - comb((N - 1) - k, M) / comb(N - 1, M)
    p_group = pe * (1 - (1 - p_s) ** L)
    expected = 1 - (1 - p_group) ** G
    assert pytest.approx(p_social(N, M, k), rel=1e-8) == expected
