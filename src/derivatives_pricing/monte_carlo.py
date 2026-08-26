"""Monte Carlo option valuation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from .black_scholes import Option, OptionType


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    paths: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        width = 1.959963984540054 * self.standard_error
        return self.price - width, self.price + width


def _payoff(underlying: np.ndarray, option: Option) -> np.ndarray:
    if option.option_type == OptionType.CALL:
        return np.maximum(underlying - option.strike, 0.0)
    return np.maximum(option.strike - underlying, 0.0)


def _result(payoffs: np.ndarray, option: Option) -> MonteCarloResult:
    discounted = exp(-option.rate * option.maturity) * payoffs
    return MonteCarloResult(
        price=float(np.mean(discounted)),
        standard_error=float(np.std(discounted, ddof=1) / sqrt(len(discounted))),
        paths=int(len(discounted)),
    )


def _normal_draws(rng: np.random.Generator, paths: int, shape: tuple[int, ...]) -> np.ndarray:
    if paths < 2:
        raise ValueError("paths must be at least two")
    half = (paths + 1) // 2
    base = rng.standard_normal((half, *shape))
    return np.concatenate([base, -base], axis=0)[:paths]


def price_european(option: Option, *, paths: int = 100_000, seed: int = 7) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    z = _normal_draws(rng, paths, ())
    terminal = option.spot * np.exp(
        (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * option.maturity
        + option.volatility * sqrt(option.maturity) * z
    )
    return _result(_payoff(terminal, option), option)


def price_asian(
    option: Option,
    *,
    paths: int = 100_000,
    steps: int = 252,
    seed: int = 7,
) -> MonteCarloResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    z = _normal_draws(rng, paths, (steps,))
    dt = option.maturity / steps
    log_returns = (
        (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
        + option.volatility * sqrt(dt) * z
    )
    prices = option.spot * np.exp(np.cumsum(log_returns, axis=1))
    return _result(_payoff(np.mean(prices, axis=1), option), option)

