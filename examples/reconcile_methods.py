from derivatives_pricing import Option, binomial_price, black_scholes, price_european

contract = Option(spot=100, strike=100, maturity=1, rate=0.05, volatility=0.20)
analytical = black_scholes(contract).price
lattice = binomial_price(contract, steps=1_000)
simulation = price_european(contract, paths=500_000, seed=42)

print(f"Black-Scholes: {analytical:.6f}")
print(f"CRR lattice:   {lattice:.6f}")
print(f"Monte Carlo:   {simulation.price:.6f} ± {1.96 * simulation.standard_error:.6f}")
