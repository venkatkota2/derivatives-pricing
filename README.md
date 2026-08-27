# derivatives-pricing

A transparent numerical-finance reference library for derivatives and fixed-income valuation across analytical, lattice, and simulation methods.

The project emphasizes reconciliation. European options can be compared across Black–Scholes, Cox–Ross–Rubinstein, and Monte Carlo implementations; bond analytics are checked against discounted cash-flow identities and pull-to-par behaviour.

## Coverage

### Options

- Black–Scholes prices and delta, gamma, vega, theta, and rho.
- Implied volatility with bracket validation and bisection.
- European and American options on a CRR binomial lattice.
- European and arithmetic-average Asian Monte Carlo pricing.
- Antithetic sampling with uncertainty estimated from independent pair averages.

### Fixed income

- Log-linear discount curves and forward rates, including valid negative-rate curves.
- Coupon and zero-coupon bond pricing.
- Yield-to-maturity solving.
- Macaulay duration, modified duration, and convexity.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
price-instrument option --spot 100 --strike 105 --volatility 0.20
price-instrument bond --face 1000 --coupon 0.05 --maturity 7 --yield-rate 0.045
```

```python
from derivatives_pricing import Option, black_scholes, binomial_price

contract = Option(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.20)
analytical = black_scholes(contract)
lattice = binomial_price(contract, steps=500)
print(analytical.price, lattice)
```

## Validation approach

Tests cover put–call parity, lattice convergence, implied-volatility recovery, pair-level Monte Carlo uncertainty, coupon-versus-yield behaviour, negative-rate curves, finite-input validation, and root-solver bracketing. Edge cases fail explicitly instead of returning plausible-looking values.

The central reconciliation is:

```text
Black–Scholes ↔ CRR lattice ↔ Monte Carlo
coupon/yield identities ↔ discount curves ↔ duration/convexity
```

For antithetic Monte Carlo, `paths` is the number of simulated paths and
`effective_samples` is the number of independent pair averages used for the
reported standard error.

## Repository layout

```text
src/derivatives_pricing/  reference valuation implementations
examples/                 reproducible cross-method reconciliation
tests/                    numerical identities and regression tests
```

## Scope

The models use deterministic rates and volatility and do not include calibration, dividends by schedule, credit migration, callable bonds, or market-data ingestion. The library is for education and model-development demonstrations, not investment decisions.
