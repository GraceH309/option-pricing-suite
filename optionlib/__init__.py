"""Option pricing & derivatives analytics toolkit.

A self-contained library combining quant-domain knowledge (equity derivatives)
with rigorous numerical methods (closed-form, Monte Carlo with variance
reduction, binomial trees, regression-based American pricing, root-finding).

Modules
-------
pricing     : Black-Scholes analytics + Greeks
montecarlo : MC European (antithetic + control variate) and American (LSM)
binomial   : Cox-Ross-Rubinstein tree (European / American)
volatility : implied-volatility solver (bisection / Newton)
"""

from . import pricing
from . import montecarlo
from . import binomial
from . import volatility

__all__ = ["pricing", "montecarlo", "binomial", "volatility"]
