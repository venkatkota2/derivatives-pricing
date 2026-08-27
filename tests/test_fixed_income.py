import numpy as np
import pytest

from derivatives_pricing import Bond, DiscountCurve, bond_analytics, bond_price, yield_to_maturity


def test_coupon_equal_to_yield_prices_at_par():
    bond = Bond(face=1_000, coupon_rate=0.05, maturity=10, frequency=2)
    assert abs(bond_price(bond, 0.05) - 1_000) < 1e-10


def test_yield_solver_and_risk_measures():
    bond = Bond(face=1_000, coupon_rate=0.04, maturity=7, frequency=2)
    price = bond_price(bond, 0.052)
    solved = yield_to_maturity(bond, price)
    analytics = bond_analytics(bond, solved)

    assert abs(solved - 0.052) < 1e-10
    assert analytics.price == pytest.approx(price)
    assert analytics.modified_duration > 0
    assert analytics.convexity > 0


def test_discount_curve_interpolates_log_discount_factors():
    curve = DiscountCurve.from_zero_rates(
        np.array([1.0, 2.0, 5.0]),
        np.array([0.03, 0.035, 0.04]),
    )
    assert curve.discount(0) == 1.0
    assert curve.discount(4) < curve.discount(3) < curve.discount(2)
    assert curve.forward_rate(1, 2) > 0


def test_negative_rate_discount_curve_is_supported():
    curve = DiscountCurve.from_zero_rates(
        np.array([1.0, 2.0, 5.0]),
        np.array([-0.01, -0.008, -0.004]),
    )

    assert np.all(curve.discount_factors > 1.0)
    assert curve.discount(2.0) == pytest.approx(np.exp(0.016))
    assert curve.forward_rate(1.0, 2.0) < 0.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_curve_and_bond_reject_non_finite_inputs(invalid):
    with pytest.raises(ValueError):
        DiscountCurve(np.array([1.0, invalid]), np.array([0.98, 0.95]))
    with pytest.raises(ValueError):
        Bond(face=1_000, coupon_rate=0.04, maturity=invalid)
    bond = Bond(face=1_000, coupon_rate=0.04, maturity=5)
    with pytest.raises(ValueError):
        bond_price(bond, invalid)
    with pytest.raises(ValueError):
        bond_analytics(bond, invalid)


def test_bond_yield_validation_is_consistent():
    bond = Bond(face=1_000, coupon_rate=0.04, maturity=5, frequency=2)
    for function in (bond_price, bond_analytics):
        with pytest.raises(ValueError, match="invalid discount factors"):
            function(bond, -2.0)


def test_yield_solver_requires_a_bracket_and_convergence():
    bond = Bond(face=1_000, coupon_rate=0.04, maturity=5, frequency=2)
    with pytest.raises(ValueError, match="cannot be bracketed"):
        yield_to_maturity(bond, 1e-300)
    with pytest.raises(ValueError, match="solver configuration"):
        yield_to_maturity(bond, 1_000, tolerance=float("nan"))
    with pytest.raises(RuntimeError, match="did not converge"):
        yield_to_maturity(bond, 900, tolerance=1e-20, max_iterations=1)
