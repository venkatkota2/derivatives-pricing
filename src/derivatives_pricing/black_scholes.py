"""Black-Scholes valuation and implied volatility."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import erf, exp, isfinite, log, pi, sqrt


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = OptionType.CALL

    def __post_init__(self) -> None:
        numeric_values = (
            self.spot,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            self.dividend_yield,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("option inputs must be finite")
        if self.spot <= 0 or self.strike <= 0 or self.maturity <= 0:
            raise ValueError("spot, strike, and maturity must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if not isinstance(self.option_type, OptionType):
            raise ValueError("option_type must be an OptionType")


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _intrinsic_present_value(option: Option) -> float:
    forward_spot = option.spot * exp((option.rate - option.dividend_yield) * option.maturity)
    if option.option_type == OptionType.CALL:
        payoff = max(forward_spot - option.strike, 0.0)
    else:
        payoff = max(option.strike - forward_spot, 0.0)
    return exp(-option.rate * option.maturity) * payoff


def black_scholes(option: Option) -> Greeks:
    """Return price and standard Black-Scholes Greeks for a European option."""
    if option.volatility == 0:
        return Greeks(
            _intrinsic_present_value(option), float("nan"), 0.0, 0.0, float("nan"), float("nan")
        )

    root_time = sqrt(option.maturity)
    d1 = (
        log(option.spot / option.strike)
        + (option.rate - option.dividend_yield + 0.5 * option.volatility * option.volatility)
        * option.maturity
    ) / (option.volatility * root_time)
    d2 = d1 - option.volatility * root_time
    discounted_spot = option.spot * exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * exp(-option.rate * option.maturity)
    density = _pdf(d1)

    if option.option_type == OptionType.CALL:
        price = discounted_spot * _cdf(d1) - discounted_strike * _cdf(d2)
        delta = exp(-option.dividend_yield * option.maturity) * _cdf(d1)
        theta = (
            -discounted_spot * density * option.volatility / (2.0 * root_time)
            - option.rate * discounted_strike * _cdf(d2)
            + option.dividend_yield * discounted_spot * _cdf(d1)
        )
        rho = option.maturity * discounted_strike * _cdf(d2)
    else:
        price = discounted_strike * _cdf(-d2) - discounted_spot * _cdf(-d1)
        delta = exp(-option.dividend_yield * option.maturity) * (_cdf(d1) - 1.0)
        theta = (
            -discounted_spot * density * option.volatility / (2.0 * root_time)
            + option.rate * discounted_strike * _cdf(-d2)
            - option.dividend_yield * discounted_spot * _cdf(-d1)
        )
        rho = -option.maturity * discounted_strike * _cdf(-d2)

    gamma = (
        exp(-option.dividend_yield * option.maturity)
        * density
        / (option.spot * option.volatility * root_time)
    )
    vega = discounted_spot * density * root_time
    return Greeks(price, delta, gamma, vega, theta, rho)


def implied_volatility(
    market_price: float,
    option: Option,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Recover volatility with a bracketed bisection solver."""
    if (
        not isfinite(market_price)
        or not isfinite(tolerance)
        or market_price < 0
        or tolerance <= 0
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("invalid solver input")
    lower, upper = 1e-8, 5.0
    lower_price = black_scholes(replace(option, volatility=lower)).price
    upper_price = black_scholes(replace(option, volatility=upper)).price
    if not lower_price - tolerance <= market_price <= upper_price + tolerance:
        raise ValueError("market price violates the solver's no-arbitrage bracket")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        price = black_scholes(replace(option, volatility=midpoint)).price
        if abs(price - market_price) <= tolerance:
            return midpoint
        if price < market_price:
            lower = midpoint
        else:
            upper = midpoint
    raise RuntimeError("implied-volatility solver did not converge")
