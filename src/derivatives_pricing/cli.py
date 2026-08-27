from __future__ import annotations

import argparse

from .black_scholes import Option, OptionType, black_scholes
from .fixed_income import Bond, bond_analytics


def main() -> None:
    parser = argparse.ArgumentParser(description="Price an option or bond")
    subparsers = parser.add_subparsers(dest="instrument", required=True)

    option_parser = subparsers.add_parser("option")
    option_parser.add_argument("--spot", type=float, required=True)
    option_parser.add_argument("--strike", type=float, required=True)
    option_parser.add_argument("--maturity", type=float, default=1.0)
    option_parser.add_argument("--rate", type=float, default=0.05)
    option_parser.add_argument("--volatility", type=float, required=True)
    option_parser.add_argument("--type", choices=["call", "put"], default="call")

    bond_parser = subparsers.add_parser("bond")
    bond_parser.add_argument("--face", type=float, required=True)
    bond_parser.add_argument("--coupon", type=float, required=True)
    bond_parser.add_argument("--maturity", type=float, required=True)
    bond_parser.add_argument("--yield-rate", type=float, required=True)
    bond_parser.add_argument("--frequency", type=int, default=2)

    args = parser.parse_args()
    if args.instrument == "option":
        result = black_scholes(
            Option(
                args.spot,
                args.strike,
                args.maturity,
                args.rate,
                args.volatility,
                option_type=OptionType(args.type),
            )
        )
        print(
            f"price={result.price:.6f} delta={result.delta:.6f} "
            f"gamma={result.gamma:.6f} vega={result.vega:.6f}"
        )
    else:
        result = bond_analytics(
            Bond(args.face, args.coupon, args.maturity, args.frequency),
            args.yield_rate,
        )
        print(
            f"price={result.price:.6f} modified_duration={result.modified_duration:.6f} "
            f"convexity={result.convexity:.6f}"
        )


if __name__ == "__main__":
    main()
