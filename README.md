# Option pricing toy toolkit

> I work at a quant shop and wanted to re-implement the pricing methods I use
> day-to-day by hand, partly to review numerical methods.
> Goal: **zero external dependencies** — only numpy + stdlib.

## Methods

| Method | File | Notes |
|--------|------|-------|
| Black-Scholes analytic | `pricing.py` | hand-rolled `_norm_cdf`, no scipy |
| Monte Carlo (European) | `montecarlo.py` | antithetic + control variate (variance reduction) |
| Longstaff-Schwartz (American) | `montecarlo.py` | quadratic basis for early exercise |
| CRR binomial tree | `binomial.py` | European + American |
| Implied volatility | `volatility.py` | bisection + Newton-Raphson |

## Why I wrote my own norm_cdf

I could've just done `from scipy.stats import norm`, but for zero deps I
implemented the Abramowitz & Stegun 26.2.17 approximation myself. Accuracy
~1e-7, good enough for pricing. (see the comment on `_norm_cdf` in `pricing.py`)

## Run it

```bash
pip install -r requirements.txt
python run_demo.py   # cross-checks every method
pytest tests/ -q     # 10 tests, all numerical-consistency checks
```

## TODO

- [ ] Want to add path-dependent options (Asian, barrier)
- [ ] Greeks finite-difference tolerance of 1e-2 is a bit loose; want to tighten to 1e-4
- [ ] American LSM basis is only quadratic right now; curious about higher orders

## Known rough edges

- In the American LSM, `itm.sum() >= 5` is a made-up threshold — when too few
  paths are in-the-money the regression gets unstable. If asked in an interview
  I need to be able to justify the 5.
- Binomial and implied-vol code is the oldest; style is rougher than the rest. Left as-is for now.

---

*Pure practice + review, not a trading system. The method table in this README is my own notes, not copied from a book.*
