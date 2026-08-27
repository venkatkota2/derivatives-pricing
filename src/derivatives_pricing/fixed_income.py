"""Fixed-income valuation and risk measures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class Bond:
    face: float
    coupon_rate: float
    maturity: float
    frequency: int = 2

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.face, self.coupon_rate, self.maturity)):
            raise ValueError("bond inputs must be finite")
        if self.face <= 0 or self.coupon_rate < 0 or self.maturity <= 0:
            raise ValueError("invalid bond terms")
        if (
            not isinstance(self.frequency, int)
            or isinstance(self.frequency, bool)
            or self.frequency <= 0
        ):
            raise ValueError("frequency must be a positive integer")
        if abs(self.maturity * self.frequency - round(self.maturity * self.frequency)) > 1e-10:
            raise ValueError("maturity must contain a whole number of coupon periods")

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        periods = round(self.maturity * self.frequency)
        times = np.arange(1, periods + 1, dtype=float) / self.frequency
        amounts = np.full(periods, self.face * self.coupon_rate / self.frequency)
        amounts[-1] += self.face
        return times, amounts


@dataclass(frozen=True)
class BondAnalytics:
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float


def _validate_yield(bond: Bond, yield_rate: float) -> float:
    if not isfinite(yield_rate):
        raise ValueError("yield must be finite")
    periodic_yield = yield_rate / bond.frequency
    if 1.0 + periodic_yield <= 0.0:
        raise ValueError("yield produces invalid discount factors")
    return periodic_yield


def bond_price(bond: Bond, yield_rate: float) -> float:
    periodic_yield = _validate_yield(bond, yield_rate)
    times, amounts = bond.cashflows()
    periods = times * bond.frequency
    discounts = np.power(1.0 + periodic_yield, -periods)
    price = float(np.sum(amounts * discounts))
    if not isfinite(price):
        raise OverflowError("bond price is not finite for the supplied yield")
    return price


def yield_to_maturity(
    bond: Bond,
    market_price: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
) -> float:
    if (
        not isfinite(market_price)
        or not isfinite(tolerance)
        or market_price <= 0
        or tolerance <= 0
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        if market_price <= 0:
            raise ValueError("market price must be positive")
        raise ValueError("invalid solver configuration")
    lower = (-1.0 + 1e-12) * bond.frequency
    lower_value = bond_price(bond, lower)
    if not isfinite(lower_value) or lower_value < market_price:
        raise ValueError("market price cannot be bracketed by valid yields")

    upper = 1.0
    upper_value = bond_price(bond, upper)
    while upper_value > market_price and upper < 1_000_000.0:
        upper *= 2.0
        upper_value = bond_price(bond, upper)
    if upper_value > market_price:
        raise ValueError("market price cannot be bracketed within supported yields")

    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        value = bond_price(bond, midpoint)
        if abs(value - market_price) <= tolerance:
            return midpoint
        if value > market_price:
            lower = midpoint
        else:
            upper = midpoint
    raise RuntimeError("yield-to-maturity solver did not converge")


def bond_analytics(bond: Bond, yield_rate: float) -> BondAnalytics:
    periodic_yield = _validate_yield(bond, yield_rate)
    times, amounts = bond.cashflows()
    periods = times * bond.frequency
    discounts = np.power(1.0 + periodic_yield, -periods)
    present_values = amounts * discounts
    price = float(np.sum(present_values))
    macaulay = float(np.sum(times * present_values) / price)
    modified = macaulay / (1.0 + periodic_yield)
    convexity = float(
        np.sum(amounts * periods * (periods + 1.0) * np.power(1.0 + periodic_yield, -periods - 2.0))
        / (price * bond.frequency**2)
    )
    if not all(isfinite(value) for value in (price, macaulay, modified, convexity)):
        raise OverflowError("bond analytics are not finite for the supplied yield")
    return BondAnalytics(price, macaulay, modified, convexity)
