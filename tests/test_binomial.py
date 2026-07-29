from optionlib import pricing, binomial


def test_crr_european_converges_to_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = pricing.bs_call(S, K, T, r, sigma)
    crr = binomial.crr_price(S, K, T, r, sigma, n_steps=400, option="call", american=False)
    assert abs(crr - bs) < 0.05


def test_crr_american_put_ge_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    euro = pricing.bs_put(S, K, T, r, sigma)
    amer = binomial.crr_price(S, K, T, r, sigma, n_steps=200, option="put", american=True)
    assert amer >= euro - 1e-9
    assert amer > euro  # early-exercise premium
