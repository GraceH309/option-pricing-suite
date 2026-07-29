# Option Pricing & Derivatives Analytics Suite

A self-contained Python library that prices equity options with **four independent
numerical methods** and cross-checks them — built to demonstrate rigorous
quant-domain + CS ability (numerical methods, Monte Carlo with variance
reduction, dynamic-programming-style American pricing, and root-finding).

> Personal project by a quant (Huaxin Capital) formalizing CS fundamentals for grad study.

## What it does

| Method | Module | Highlights |
|---|---|---|
| Black-Scholes analytics | `pricing.py` | Closed-form call/put price + 5 Greeks (delta/gamma/vega/theta/rho) |
| Monte Carlo (European) | `montecarlo.py` | GBM terminal simulation with **antithetic variates + control variate** |
| Longstaff-Schwartz (American) | `montecarlo.py` | Least-squares Monte Carlo with quadratic basis for early-exercise |
| CRR binomial tree | `binomial.py` | Cox-Ross-Rubinstein lattice, European **and** American |
| Implied vol solver | `volatility.py` | Bisection + Newton-Raphson recovery of σ from a market price |

All methods converge to the same price on a plain-vanilla option, and the
American premium is correctly positive vs its European counterpart.

## Run it

```bash
pip install -r requirements.txt
python run_demo.py        # cross-check all methods on one ATM option
pytest tests/ -q          # 13 tests, all numerical-consistency checks
```

## Layout

```
optionlib/
  pricing.py      BS analytics + Greeks
  montecarlo.py   MC European (variance-reduced) + LSM American
  binomial.py     CRR tree (EU / US)
  volatility.py   implied-vol solver
tests/            pytest suite (BS vs MC vs tree, Greeks vs finite diff, IV recovery)
run_demo.py       one-command cross-check
```

## A note on integrity

This is original code written to *demonstrate* methods I use daily. Every result
is reproducible (fixed RNG seeds) and verified by the test suite — read the source
and the tests before citing it anywhere.
