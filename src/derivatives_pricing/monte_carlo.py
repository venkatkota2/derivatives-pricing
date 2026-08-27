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
    effective_samples: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        width = 1.959963984540054 * self.standard_error
        return self.price - width, self.price + width


def _payoff(underlying: np.ndarray, option: Option) -> np.ndarray:
    if option.option_type == OptionType.CALL:
        return np.maximum(underlying - option.strike, 0.0)
    return np.maximum(option.strike - underlying, 0.0)


def _result(pair_estimates: np.ndarray, option: Option, *, paths: int) -> MonteCarloResult:
    """Summarize independent antithetic-pair averages.

    Each pair average is one independent sampling unit. Treating the two
    payoffs within a pair as IID would understate or overstate uncertainty.
    """
    discounted = exp(-option.rate * option.maturity) * pair_estimates
    if not np.all(np.isfinite(discounted)):
        raise OverflowError("simulation produced non-finite discounted payoffs")
    return MonteCarloResult(
        price=float(np.mean(discounted)),
        standard_error=float(np.std(discounted, ddof=1) / sqrt(len(discounted))),
        paths=paths,
        effective_samples=len(discounted),
    )


def _normal_pairs(rng: np.random.Generator, paths: int, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(paths, int) or isinstance(paths, bool) or paths < 4 or paths % 2:
        raise ValueError("antithetic simulation requires an even path count of at least four")
    return rng.standard_normal((paths // 2, *shape))


def price_european(option: Option, *, paths: int = 100_000, seed: int = 7) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    z = _normal_pairs(rng, paths, ())
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * option.maturity
    diffusion = option.volatility * sqrt(option.maturity) * z
    terminal_positive = option.spot * np.exp(drift + diffusion)
    terminal_negative = option.spot * np.exp(drift - diffusion)
    pair_estimates = 0.5 * (_payoff(terminal_positive, option) + _payoff(terminal_negative, option))
    return _result(pair_estimates, option, paths=paths)


def price_asian(
    option: Option,
    *,
    paths: int = 100_000,
    steps: int = 252,
    seed: int = 7,
) -> MonteCarloResult:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    rng = np.random.default_rng(seed)
    z = _normal_pairs(rng, paths, (steps,))
    dt = option.maturity / steps
    drift = (option.rate - option.dividend_yield - 0.5 * option.volatility**2) * dt
    diffusion = option.volatility * sqrt(dt) * z
    prices_positive = option.spot * np.exp(np.cumsum(drift + diffusion, axis=1))
    prices_negative = option.spot * np.exp(np.cumsum(drift - diffusion, axis=1))
    pair_estimates = 0.5 * (
        _payoff(np.mean(prices_positive, axis=1), option)
        + _payoff(np.mean(prices_negative, axis=1), option)
    )
    return _result(pair_estimates, option, paths=paths)
