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


def test_non_monotone_discount_curve_is_rejected():
    with pytest.raises(ValueError):
        DiscountCurve(np.array([1.0, 2.0]), np.array([0.95, 0.97]))

