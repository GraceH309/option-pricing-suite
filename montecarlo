"""Monte Carlo option pricing with variance reduction.

- ``mc_european_price`` : European options via terminal-simulation GBM, with
  antithetic variates and a control variate (terminal stock) to shrink error.
- ``mc_american_lsm``    : American options via Longstaff-Schwartz (least-squares
  Monte Carlo) using a quadratic basis in the simulated stock price.
"""

import numpy as np

from .pricing import bs_price


def _simulate_terminal(S, T, r, sigma, q, n_paths, seed, antithetic):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    mu = (r - q - 0.5 * sigma * sigma) * T
    sd = sigma * np.sqrt(T)
    ST = S * np.exp(mu + sd * z)
    if antithetic:
        ST = np.concatenate([ST, S * np.exp(mu - sd * z)])
    return ST


def mc_european_price(
    S, K, T, r, sigma, n_paths=200_000, option="call", q=0.0,
    seed=None, antithetic=True, control_variate=True,
):
    """Price a European option by Monte Carlo.

    Returns ``{"price": float, "stderr": float}``.  ``stderr`` is the standard
    error of the Monte Carlo estimator (no-bias guarantee when control variate
    is used, since E[control] = 0).
    """
    if T <= 0:
        payoff = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return {"price": float(payoff), "stderr": 0.0}

    ST = _simulate_terminal(S, T, r, sigma, q, n_paths, seed, antithetic)
    if option == "call":
        pay = np.maximum(ST - K, 0.0)
    else:
        pay = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * T)
    Y = disc * pay

    if control_variate:
        # Control variate: terminal stock has known expectation S*exp((r-q)T).
        X = ST
        EX = S * np.exp((r - q) * T)
        cov = np.cov(Y, X)
        b = cov[0, 1] / cov[1, 1]
        Y = Y - b * (X - EX)

    return {"price": float(Y.mean()), "stderr": float(Y.std(ddof=1) / np.sqrt(len(Y)))}


def mc_american_lsm(
    S, K, T, r, sigma, n_paths=30_000, n_steps=50, option="put", q=0.0, seed=None
):
    """Price an American option via Longstaff-Schwartz least-squares Monte Carlo.

    At each exercise date we regress the *continuation value* (discounted
    next-step cashflow) on a quadratic basis of the simulated stock price, then
    exercise early when the immediate exercise value exceeds that estimate.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diff = sigma * np.sqrt(dt)
    log_paths = np.log(S) + np.cumsum(drift + diff * Z, axis=1)
    paths = np.exp(np.column_stack([np.full(n_paths, np.log(S)), log_paths]))

    if option == "call":
        exer = np.maximum(paths - K, 0.0)
    else:
        exer = np.maximum(K - paths, 0.0)

    disc = np.exp(-r * dt)
    cf = exer.copy()

    for t in range(n_steps - 1, -1, -1):
        cont_next = cf[:, t + 1] * disc
        cf[:, t] = cont_next  # default: hold (continue)
        itm = exer[:, t] > 0
        if itm.sum() >= 5:
            X = paths[itm, t]
            Y = cont_next[itm]
            Xc = X - X.mean()
            M = np.column_stack([np.ones_like(Xc), Xc, Xc * Xc])
            coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
            cont_est = coef[0] + coef[1] * Xc + coef[2] * Xc * Xc
            early = exer[itm, t] > cont_est
            cf[itm, t] = np.where(early, exer[itm, t], cont_next[itm])
        # out-of-the-money paths keep cont_next (already assigned above)

    return float(cf[:, 0].mean())
