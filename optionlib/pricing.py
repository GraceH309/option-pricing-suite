"""Black-Scholes analytical pricing and Greeks.

All functions use the standard GBM log-normal model with continuous dividend
yield ``q``.  Time ``T`` is measured in years; rates/vols are annualized.
"""

import math

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard normal density phi(x)."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the Abramowitz & Stegun 26.2.17 approximation."""
    sign = 1.0 if x >= 0 else -1.0
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = (
        0.319381530 * t
        - 0.356563782 * t * t
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    c = 1.0 - _norm_pdf(z) * poly
    return c if x >= 0 else 1.0 - c


def _d1_d2(S, K, T, r, sigma, q=0.0):
    sqrtT = math.sqrt(T)
    if sigma <= 0 or T <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_call(S, K, T, r, sigma, q=0.0) -> float:
    """Black-Scholes price of a European call."""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0) -> float:
    """Black-Scholes price of a European put."""
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_price(S, K, T, r, sigma, q=0.0, option: str = "call") -> float:
    """Dispatch to call/put Black-Scholes price."""
    option = option.lower()
    if option == "call":
        return bs_call(S, K, T, r, sigma, q)
    if option == "put":
        return bs_put(S, K, T, r, sigma, q)
    raise ValueError("option must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, q=0.0, option: str = "call") -> dict:
    """Black-Scholes Greeks as a dictionary.

    Vega is reported per 1.00 change in volatility (absolute), so divide by 100
    for the common "per 1% vol" convention.  Theta is dV/dT (per year).
    """
    option = option.lower()
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    sqrtT = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    gamma = disc_q * nd1 / (S * sigma * sqrtT)
    vega = S * disc_q * nd1 * sqrtT

    if option == "call":
        delta = disc_q * Nd1
        theta = (
            -S * disc_q * nd1 * sigma / (2.0 * sqrtT)
            - r * K * disc_r * Nd2
            + q * S * disc_q * Nd1
        )
        rho = K * T * disc_r * Nd2
    elif option == "put":
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -S * disc_q * nd1 * sigma / (2.0 * sqrtT)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )
        rho = -K * T * disc_r * _norm_cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
