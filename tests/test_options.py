from dataclasses import replace
from math import exp, sqrt

import numpy as np
import pytest

from derivatives_pricing import (
    Option,
    OptionType,
    binomial_price,
    black_scholes,
    implied_volatility,
    price_asian,
    price_european,
)


def test_put_call_parity_and_lattice_convergence():
    call = Option(100, 105, 1.5, 0.04, 0.22, 0.01, OptionType.CALL)
    put = replace(call, option_type=OptionType.PUT)
    call_price = black_scholes(call).price
    put_price = black_scholes(put).price
    parity = 100 * exp(-0.01 * 1.5) - 105 * exp(-0.04 * 1.5)

    assert abs(call_price - put_price - parity) < 1e-10
    assert abs(binomial_price(call, steps=1_000) - call_price) < 0.02


def test_implied_volatility_recovers_input():
    option = Option(100, 100, 2, 0.03, 0.31)
    market_price = black_scholes(option).price
    assert abs(implied_volatility(market_price, replace(option, volatility=0.15)) - 0.31) < 1e-8


def test_antithetic_european_uncertainty_uses_pair_averages():
    option = Option(100, 100, 1, 0.05, 0.20)
    paths = 20
    seed = 11
    result = price_european(option, paths=paths, seed=seed)

    z = np.random.default_rng(seed).standard_normal(paths // 2)
    drift = (option.rate - 0.5 * option.volatility**2) * option.maturity
    diffusion = option.volatility * sqrt(option.maturity) * z
    positive = np.maximum(option.spot * np.exp(drift + diffusion) - option.strike, 0.0)
    negative = np.maximum(option.spot * np.exp(drift - diffusion) - option.strike, 0.0)
    discounted_pairs = exp(-option.rate * option.maturity) * 0.5 * (positive + negative)

    assert result.paths == paths
    assert result.effective_samples == paths // 2
    assert result.price == pytest.approx(np.mean(discounted_pairs))
    assert result.standard_error == pytest.approx(
        np.std(discounted_pairs, ddof=1) / sqrt(paths // 2)
    )


def test_antithetic_asian_uncertainty_uses_pair_averages():
    option = Option(100, 100, 1, 0.03, 0.25)
    paths, steps, seed = 12, 4, 19
    result = price_asian(option, paths=paths, steps=steps, seed=seed)

    z = np.random.default_rng(seed).standard_normal((paths // 2, steps))
    dt = option.maturity / steps
    drift = (option.rate - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * sqrt(dt) * z
    positive = option.spot * np.exp(np.cumsum(drift + diffusion, axis=1))
    negative = option.spot * np.exp(np.cumsum(drift - diffusion, axis=1))
    pair_payoffs = 0.5 * (
        np.maximum(np.mean(positive, axis=1) - option.strike, 0.0)
        + np.maximum(np.mean(negative, axis=1) - option.strike, 0.0)
    )
    discounted_pairs = exp(-option.rate * option.maturity) * pair_payoffs

    assert result.effective_samples == paths // 2
    assert result.standard_error == pytest.approx(
        np.std(discounted_pairs, ddof=1) / sqrt(paths // 2)
    )


def test_antithetic_simulation_rejects_odd_path_counts():
    option = Option(100, 100, 1, 0.05, 0.20)
    with pytest.raises(ValueError, match="even path count"):
        price_european(option, paths=101)


def test_american_put_is_not_cheaper_than_european_put():
    put = Option(100, 110, 1, 0.05, 0.25, option_type=OptionType.PUT)
    european = binomial_price(put, steps=500, american=False)
    american = binomial_price(put, steps=500, american=True)
    assert american >= european


def test_zero_volatility_american_put_recognizes_immediate_exercise():
    put = Option(100, 120, 1, 0.10, 0.0, option_type=OptionType.PUT)

    european = binomial_price(put, steps=100, american=False)
    american = binomial_price(put, steps=100, american=True)

    assert american == pytest.approx(20.0)
    assert american > european


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_discrete_pricers_require_positive_integer_steps(steps):
    option = Option(100, 100, 1, 0.05, 0.20)
    with pytest.raises(ValueError, match="positive integer"):
        binomial_price(option, steps=steps)
    with pytest.raises(ValueError, match="positive integer"):
        price_asian(option, paths=20, steps=steps)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_option_and_solver_reject_non_finite_inputs(invalid):
    with pytest.raises(ValueError):
        Option(100, 100, 1, invalid, 0.2)
    option = Option(100, 100, 1, 0.05, 0.20)
    with pytest.raises(ValueError):
        implied_volatility(invalid, option)


def test_implied_volatility_reports_non_convergence():
    option = Option(100, 100, 1, 0.05, 0.20)
    market_price = black_scholes(option).price
    with pytest.raises(RuntimeError, match="did not converge"):
        implied_volatility(market_price, option, tolerance=1e-20, max_iterations=1)
