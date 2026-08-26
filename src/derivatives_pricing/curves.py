"""Discount-curve representation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DiscountCurve:
    times: NDArray[np.float64]
    discount_factors: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        factors = np.asarray(self.discount_factors, dtype=float)
        if times.ndim != 1 or factors.ndim != 1 or len(times) != len(factors):
            raise ValueError("times and discount factors must be equally sized vectors")
        if len(times) == 0 or np.any(times <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("curve times must be positive and increasing")
        if np.any(factors <= 0) or np.any(np.diff(factors) > 1e-12):
            raise ValueError("discount factors must be positive and non-increasing")

    @classmethod
    def from_zero_rates(
        cls,
        times: NDArray[np.float64],
        zero_rates: NDArray[np.float64],
    ) -> "DiscountCurve":
        times_array = np.asarray(times, dtype=float)
        rates_array = np.asarray(zero_rates, dtype=float)
        if times_array.shape != rates_array.shape:
            raise ValueError("times and rates must have the same shape")
        return cls(times_array, np.exp(-times_array * rates_array))

    def discount(self, time: float) -> float:
        if time < 0:
            raise ValueError("time must be non-negative")
        if time == 0:
            return 1.0
        points = np.concatenate([[0.0], np.asarray(self.times)])
        logs = np.concatenate([[0.0], np.log(np.asarray(self.discount_factors))])
        if time <= points[-1]:
            return float(exp(np.interp(time, points, logs)))
        slope = (logs[-1] - logs[-2]) / (points[-1] - points[-2])
        return float(exp(logs[-1] + slope * (time - points[-1])))

    def forward_rate(self, start: float, end: float) -> float:
        if start < 0 or end <= start:
            raise ValueError("forward interval must satisfy 0 <= start < end")
        return log(self.discount(start) / self.discount(end)) / (end - start)

