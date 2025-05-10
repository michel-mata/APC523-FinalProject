import pytest

from cultural_evolution.popularity import make_transition_functions


def test_transition_simple_social():
    """
    When asocial returns 0 and social is constant s,
    T_plus(N,M,k) = s*(1 - k/N),
    T_minus(N,M,k) = (1 - s)*(k/N) for k>=1, and 0 for k=0.
    """
    s = 0.2

    def asocial(k):
        return 0.0

    def social(N, M, k):
        return s

    T_plus, T_minus = make_transition_functions(asocial, social)
    N, M = 5, 3

    # Test for k from 0 to N
    for k in range(0, N + 1):
        expected_plus = s * (1 - k / N)
        assert pytest.approx(T_plus(N, M, k), rel=1e-8) == expected_plus

        if k == 0:
            assert pytest.approx(T_minus(N, M, k), abs=1e-8) == 0.0
        else:
            expected_minus = (1 - s) * (k / N)
            assert pytest.approx(T_minus(N, M, k), rel=1e-8) == expected_minus


def test_transition_simple_asocial():
    """
    When social returns 0 and asocial is a at k=0 only,
    T_plus(N,M,0) = a*(1 - 0/N) = a,
    T_plus(N,M,k>0) = 0,
    T_minus(N,M,k) = k/N
    """
    a = 0.5

    def asocial(k):
        return a if k == 0 else 0.0

    def social(N, M, k):
        return 0.0

    T_plus, T_minus = make_transition_functions(asocial, social)
    N, M = 6, 2

    # k=0 case
    assert pytest.approx(T_plus(N, M, 0), rel=1e-8) == a
    assert pytest.approx(T_minus(N, M, 0), abs=1e-8) == 0.0

    # k=1..N
    for k in range(1, N + 1):
        assert pytest.approx(T_plus(N, M, k), abs=1e-8) == 0.0
        expected_minus = k / N
        assert pytest.approx(T_minus(N, M, k), rel=1e-8) == expected_minus


def test_transition_combined():
    """
    When both asocial and social are nonzero,
    verify T_plus and T_minus against the formula.
    """

    def asocial(k):
        return 0.1 if k == 0 else 0.0

    def social(N, M, k):
        return 0.3 if k % 2 == 0 else 0.1

    T_plus, T_minus = make_transition_functions(asocial, social)
    N, M = 8, 4

    for k in range(0, N + 1):
        innovation = 0.1 if k == 0 else 0.0
        soc = 0.3 if k % 2 == 0 else 0.1
        expected_plus = (innovation + soc) * (1 - k / N)
        assert pytest.approx(T_plus(N, M, k), rel=1e-8) == expected_plus

        if k == 0:
            exp_minus = 0.0
        else:
            prev = 0.3 if (k - 1) % 2 == 0 else 0.1
            exp_minus = (1 - prev) * (k / N)
        assert pytest.approx(T_minus(N, M, k), rel=1e-8) == exp_minus
