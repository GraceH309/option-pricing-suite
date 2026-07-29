import math

from optionlib import pricing

"""Pricing-consistency tests.

Design notes (for myself, and so I can explain it in an interview):
1. Put-call parity is a hard constraint under the zero-model-assumption, so it's
   my go-to sanity check — must always hold.
2. Greeks are verified by central finite difference: h can't be too small (float
   precision) or too large (truncation error). delta/gamma stable at h=1e-2,
   vega/theta at 1e-4.
3. I memorized a Hull textbook benchmark: S=K=100, T=1, r=5%, sigma=20% -> 10.4506,
   as a backstop for the analytic formula.
"""


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    c = pricing.bs_call(S, K, T, r, sigma)
    p = pricing.bs_put(S, K, T, r, sigma)
    lhs = c - p
    rhs = S - K * math.exp(-r * T)  # q = 0
    assert abs(lhs - rhs) < 1e-9


def test_known_bs_call_value():
    # Classic textbook case: S=K=100, T=1, r=5%, sigma=20% -> ~10.4506
    c = pricing.bs_call(100.0, 100.0, 1.0, 0.05, 0.20)
    assert abs(c - 10.4506) < 1e-3


def test_greeks_match_finite_difference():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # delta: dC/dS
    h = 1e-2
    delta_fd = (pricing.bs_call(S + h, K, T, r, sigma) - pricing.bs_call(S - h, K, T, r, sigma)) / (2 * h)
    assert abs(delta_fd - pricing.greeks(S, K, T, r, sigma)["delta"]) < 1e-2

    # gamma: d^2C/dS^2
    gamma_fd = (
        pricing.bs_call(S + h, K, T, r, sigma)
        - 2 * pricing.bs_call(S, K, T, r, sigma)
        + pricing.bs_call(S - h, K, T, r, sigma)
    ) / (h * h)
    assert abs(gamma_fd - pricing.greeks(S, K, T, r, sigma)["gamma"]) < 1e-2

    # vega: dC/dsigma (per 1.00)
    hs = 1e-4
    vega_fd = (pricing.bs_call(S, K, T, r, sigma + hs) - pricing.bs_call(S, K, T, r, sigma - hs)) / (2 * hs)
    assert abs(vega_fd - pricing.greeks(S, K, T, r, sigma)["vega"]) < 1e-2

    # theta: our convention is theta = -dC/dT (time decay), negative for calls
    ht = 1e-4
    theta_fd = -(pricing.bs_call(S, K, T + ht, r, sigma) - pricing.bs_call(S, K, T - ht, r, sigma)) / (2 * ht)
    assert abs(theta_fd - pricing.greeks(S, K, T, r, sigma)["theta"]) < 1e-2
