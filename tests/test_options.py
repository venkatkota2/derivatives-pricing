from dataclasses import replace
from math import exp

from derivatives_pricing import (
    Option,
    OptionType,
    binomial_price,
    black_scholes,
    implied_volatility,
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


def test_monte_carlo_interval_contains_analytical_price():
    option = Option(100, 100, 1, 0.05, 0.20)
    analytical = black_scholes(option).price
    result = price_european(option, paths=200_000, seed=11)
    low, high = result.confidence_interval_95
    assert low <= analytical <= high


def test_american_put_is_not_cheaper_than_european_put():
    put = Option(100, 110, 1, 0.05, 0.25, option_type=OptionType.PUT)
    european = binomial_price(put, steps=500, american=False)
    american = binomial_price(put, steps=500, american=True)
    assert american >= european

