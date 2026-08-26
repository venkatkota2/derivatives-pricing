# derivatives-pricing

A tested Python library for derivatives and fixed-income valuation across analytical, lattice, and simulation methods.

The project emphasizes reconciliation. European options can be compared across Black–Scholes, Cox–Ross–Rubinstein, and Monte Carlo implementations; bond analytics are checked against discounted cash-flow identities and pull-to-par behaviour.

## Coverage

### Options

- Black–Scholes prices and delta, gamma, vega, theta, and rho.
- Implied volatility with bracket validation and bisection.
- European and American options on a CRR binomial lattice.
- European and arithmetic-average Asian Monte Carlo pricing.
- Antithetic sampling and standard-error estimates.

### Fixed income

- Log-linear discount curves and forward rates.
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

Tests cover put–call parity, lattice convergence, implied-volatility recovery, Monte Carlo confidence intervals, coupon-versus-yield behaviour, and discount-curve monotonicity. Edge cases fail explicitly instead of returning plausible-looking values.

## Scope

The models use deterministic rates and volatility and do not include calibration, dividends by schedule, credit migration, callable bonds, or market-data ingestion. The library is for education and model-development demonstrations, not investment decisions.

