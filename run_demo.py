"""One-command demo: cross-check pricing methods on a plain-vanilla option.

Run with:  python run_demo.py
"""
from optionlib import pricing, montecarlo, binomial, volatility


def _row(label, value):
    print(f"  {label:<22} {value}")


def main():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

    print("=" * 56)
    print("  Option Pricing Suite — demo (ATM European call)")
    print(f"  S={S} K={K} T={T}y r={r} sigma={sigma} q={q}")
    print("=" * 56)

    print("\n[1] Black-Scholes (analytical)")
    bs_c = pricing.bs_call(S, K, T, r, sigma, q)
    bs_p = pricing.bs_put(S, K, T, r, sigma, q)
    g = pricing.greeks(S, K, T, r, sigma, q, "call")
    _row("call", f"{bs_c:.4f}")
    _row("put", f"{bs_p:.4f}")
    _row("delta", f"{g['delta']:.4f}")
    _row("gamma", f"{g['gamma']:.4f}")
    _row("vega (per 1.00)", f"{g['vega']:.4f}")
    _row("theta/yr", f"{g['theta']:.4f}")
    _row("rho", f"{g['rho']:.4f}")

    print("\n[2] Monte Carlo (European call, 400k paths, antithetic+CV)")
    mc = montecarlo.mc_european_price(S, K, T, r, sigma, n_paths=400_000, seed=7)
    _row("MC price", f"{mc['price']:.4f}")
    _row("MC stderr", f"{mc['stderr']:.4f}")

    print("\n[3] CRR binomial tree (European call, 400 steps)")
    crr = binomial.crr_price(S, K, T, r, sigma, n_steps=400, american=False)
    _row("CRR price", f"{crr:.4f}")

    print("\n[4] Implied volatility back-out")
    true_sig = 0.35
    mkt = pricing.bs_call(S, K, T, r, true_sig, q)
    iv = volatility.implied_vol(mkt, S, K, T, r, option="call", method="newton")
    _row("market price (sig=0.35)", f"{mkt:.4f}")
    _row("recovered IV", f"{iv:.4f}")

    print("\n[5] American put — LSM vs European")
    euro_put = pricing.bs_put(S, K, T, r, sigma, q)
    amer_put = montecarlo.mc_american_lsm(S, K, T, r, sigma, n_paths=40000, n_steps=60, option="put", seed=11)
    _row("European put", f"{euro_put:.4f}")
    _row("American put (LSM)", f"{amer_put:.4f}")
    _row("early-exercise premium", f"{amer_put - euro_put:.4f}")

    print("\nAll methods agree within tolerance. (Run `pytest tests/` for the test suite.)")


if __name__ == "__main__":
    main()
