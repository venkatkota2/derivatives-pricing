"""Fixed-income valuation and risk measures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bond:
    face: float
    coupon_rate: float
    maturity: float
    frequency: int = 2

    def __post_init__(self) -> None:
        if self.face <= 0 or self.coupon_rate < 0 or self.maturity <= 0:
            raise ValueError("invalid bond terms")
        if self.frequency <= 0 or abs(self.maturity * self.frequency - round(self.maturity * self.frequency)) > 1e-10:
            raise ValueError("maturity must contain a whole number of coupon periods")

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        periods = int(round(self.maturity * self.frequency))
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


def bond_price(bond: Bond, yield_rate: float) -> float:
    if yield_rate <= -bond.frequency:
        raise ValueError("yield produces invalid discount factors")
    times, amounts = bond.cashflows()
    periods = times * bond.frequency
    discounts = np.power(1.0 + yield_rate / bond.frequency, -periods)
    return float(np.sum(amounts * discounts))


def yield_to_maturity(
    bond: Bond,
    market_price: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 300,
) -> float:
    if market_price <= 0:
        raise ValueError("market price must be positive")
    lower = -0.99 * bond.frequency
    upper = 1.0
    while bond_price(bond, upper) > market_price and upper < 100:
        upper *= 2.0
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        value = bond_price(bond, midpoint)
        if abs(value - market_price) <= tolerance:
            return midpoint
        if value > market_price:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def bond_analytics(bond: Bond, yield_rate: float) -> BondAnalytics:
    times, amounts = bond.cashflows()
    periods = times * bond.frequency
    periodic_yield = yield_rate / bond.frequency
    discounts = np.power(1.0 + periodic_yield, -periods)
    present_values = amounts * discounts
    price = float(np.sum(present_values))
    macaulay = float(np.sum(times * present_values) / price)
    modified = macaulay / (1.0 + periodic_yield)
    convexity = float(
        np.sum(
            amounts
            * periods
            * (periods + 1.0)
            * np.power(1.0 + periodic_yield, -periods - 2.0)
        )
        / (price * bond.frequency**2)
    )
    return BondAnalytics(price, macaulay, modified, convexity)

