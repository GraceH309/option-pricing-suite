import numpy as np

from optionlib import pricing, montecarlo


def test_mc_european_converges_to_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = pricing.bs_call(S, K, T, r, sigma)
    res = montecarlo.mc_european_price(S, K, T, r, sigma, n_paths=400_000, seed=7)
    # Monte Carlo within its own standard error + a small bias tolerance
    assert abs(res["price"] - bs) < 0.10
    assert res["stderr"] < 0.05


def test_antithetic_reduces_stderr():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    n = 40_000
    plain = montecarlo.mc_european_price(S, K, T, r, sigma, n_paths=n, seed=3, antithetic=False, control_variate=False)
    anti = montecarlo.mc_european_price(S, K, T, r, sigma, n_paths=n, seed=3, antithetic=True, control_variate=False)
    assert anti["stderr"] < plain["stderr"]


def test_american_put_exceeds_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    euro = pricing.bs_put(S, K, T, r, sigma)
    amer = montecarlo.mc_american_lsm(S, K, T, r, sigma, n_paths=40_000, n_steps=60, seed=11)
    # Early exercise premium must be positive for an American put.
    assert amer > euro
    # Sanity: classic value is ~6.5 for these params.
    assert 5.8 < amer < 7.5
