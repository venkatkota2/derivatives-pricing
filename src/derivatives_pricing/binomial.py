"""Cox-Ross-Rubinstein option lattice."""

from __future__ import annotations

from math import exp, sqrt

import numpy as np

from .black_scholes import Option, OptionType


def binomial_price(option: Option, *, steps: int = 500, american: bool = False) -> float:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(american, bool):
        raise ValueError("american must be a boolean")
    if option.volatility == 0:
        exercise_times = (
            np.linspace(0.0, option.maturity, steps + 1)
            if american
            else np.array([option.maturity])
        )
        deterministic_spots = option.spot * np.exp(
            (option.rate - option.dividend_yield) * exercise_times
        )
        intrinsic = (
            np.maximum(deterministic_spots - option.strike, 0.0)
            if option.option_type == OptionType.CALL
            else np.maximum(option.strike - deterministic_spots, 0.0)
        )
        values = np.exp(-option.rate * exercise_times) * intrinsic
        return float(np.max(values))

    dt = option.maturity / steps
    up = exp(option.volatility * sqrt(dt))
    down = 1.0 / up
    growth = exp((option.rate - option.dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("lattice parameters imply an invalid risk-neutral probability")
    discount = exp(-option.rate * dt)

    nodes = np.arange(steps + 1)
    spots = option.spot * np.power(up, nodes) * np.power(down, steps - nodes)
    if option.option_type == OptionType.CALL:
        values = np.maximum(spots - option.strike, 0.0)
    else:
        values = np.maximum(option.strike - spots, 0.0)

    for level in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
        if american:
            nodes = np.arange(level + 1)
            spots = option.spot * np.power(up, nodes) * np.power(down, level - nodes)
            intrinsic = (
                np.maximum(spots - option.strike, 0.0)
                if option.option_type == OptionType.CALL
                else np.maximum(option.strike - spots, 0.0)
            )
            values = np.maximum(values, intrinsic)
    return float(values[0])
