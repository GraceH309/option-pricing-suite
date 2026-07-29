"""Cox-Ross-Rubinstein (CRR) binomial-tree option pricing.

Supports European and American exercise, call/put, with continuous dividend
yield.  Converges to Black-Scholes as ``n_steps`` grows.
"""

import math

import numpy as np


def crr_price(S, K, T, r, sigma, n_steps=100, option="call", american=True, q=0.0):
    """Price an option with a CRR binomial lattice."""
    if T <= 0:
        if option == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(n_steps + 1)
    St = S * (u**j) * (d ** (n_steps - j))
    if option == "call":
        v = np.maximum(St - K, 0.0)
    else:
        v = np.maximum(K - St, 0.0)

    for i in range(n_steps - 1, -1, -1):
        v = disc * (p * v[1:] + (1.0 - p) * v[:-1])
        if american:
            St_i = S * (u ** j[: i + 1]) * (d ** (i - j[: i + 1]))
            if option == "call":
                ev = np.maximum(St_i - K, 0.0)
            else:
                ev = np.maximum(K - St_i, 0.0)
            v = np.maximum(v, ev)

    return float(v[0])
