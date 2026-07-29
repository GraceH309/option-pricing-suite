# option-pricing-suite

Code I wrote to actually understand option pricing instead of just calling a
library. I'm in quant, so Black-Scholes is something I use but had never
re-derived from scratch — this closed that gap.

## what's in here

- `pricing.py` — Black-Scholes call/put + Greeks. I hand-rolled `_norm_cdf`
  (Abramowitz & Stegun 26.2.17) instead of importing scipy, because I wanted
  zero external deps and the approximation is good to ~1e-7 anyway. That part
  I'm happy with.
- `montecarlo.py` — European MC with antithetic + control variates. The American
  side uses Longstaff-Schwartz; the `itm.sum() >= 5` threshold for when to run
  the regression was a guess — fewer paths and the beta estimates get noisy, but
  I never properly tuned it, just picked 5 and moved on.
- `binomial.py` — CRR tree. European + American. Oldest code in here, looks it.
- `volatility.py` — implied vol via bisection + Newton. Newton diverges when
  vega is near zero, so bisection is the one I actually trust.

`run_demo.py` runs all of it and cross-checks the methods against each other.
`pytest tests/` — 10 tests, mostly put-call parity and finite-difference Greeks.

## the numpy thing

numpy 2.x shipped and broke my original code (some array-indexing change). I
pinned `numpy<2` in requirements rather than rewrite at midnight — should fix
properly later, it's a TODO I keep skipping.

## didn't finish

Path-dependent options (Asian, Barrier). I started an Asian MC with
Brownian-bridge conditioning for the variance reduction and couldn't get it to
behave, so it's sitting dead in `experiments/`. Maybe revisit before onsites.

## run

    pip install -r requirements.txt
    python run_demo.py
    pytest tests/ -q

Practice repo, not a trading tool. Don't point it at a real book.
