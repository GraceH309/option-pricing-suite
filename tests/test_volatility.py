from optionlib import pricing, volatility


def test_implied_vol_recovers_input():
    S, K, T, r, sigma_true = 100.0, 100.0, 1.0, 0.05, 0.35
    mkt = pricing.bs_call(S, K, T, r, sigma_true)
    iv_bisect = volatility.implied_vol(mkt, S, K, T, r, option="call", method="bisection")
    iv_newton = volatility.implied_vol(mkt, S, K, T, r, option="call", method="newton")
    assert abs(iv_bisect - sigma_true) < 1e-4
    assert abs(iv_newton - sigma_true) < 1e-4


def test_implied_vol_consistency_call_put():
    S, K, T, r, sig = 100.0, 105.0, 0.5, 0.03, 0.25
    mkt_call = pricing.bs_call(S, K, T, r, sig)
    mkt_put = pricing.bs_put(S, K, T, r, sig)
    iv_c = volatility.implied_vol(mkt_call, S, K, T, r, option="call")
    iv_p = volatility.implied_vol(mkt_put, S, K, T, r, option="put")
    assert abs(iv_c - iv_p) < 1e-3
