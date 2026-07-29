"""Implied-volatility solver.

Recover the Black-Scholes volatility that reproduces a market option price.
Two methods: a robust bisection and a faster Newton-Raphson using Vega.
"""

from .pricing import bs_price, greeks


def implied_vol(
    market_price, S, K, T, r, q=0.0, option="call", method="bisection", tol=1e-8, max_iter=200
):
    """Return the implied volatility (annualized) for ``market_price``."""
    if market_price <= 0:
        raise ValueError("market_price must be > 0")
    option = option.lower()

    if method == "newton":
        sigma = 0.20
        for _ in range(max_iter):
            pv = bs_price(S, K, T, r, sigma, q, option)
            vega = greeks(S, K, T, r, sigma, q, option)["vega"]
            if vega == 0:
                break
            step = (pv - market_price) / vega
            if abs(step) < tol:
                break
            sigma -= step
            if sigma <= 1e-6:
                sigma = 1e-6
        return float(sigma)

    # bisection
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pv = bs_price(S, K, T, r, mid, q, option)
        if pv > market_price:
            hi = mid
        else:
            lo = mid
        if abs(pv - market_price) < tol:
            break
    return 0.5 * (lo + hi)
