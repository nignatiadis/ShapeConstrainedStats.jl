# ShapeConstrainedStats.jl


[Build Status](https://github.com/nignatiadis/ShapeConstrainedStats.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nignatiadis/ShapeConstrainedStats.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nignatiadis/ShapeConstrainedStats.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nignatiadis/ShapeConstrainedStats.jl)


A Julia package for nonparametric shape-restricted statistics, see e.g. [1] for a recent review. Currently only implements the `IsotonicRegression` type as a `StatsBase.RegressionModel`. The main fitting function was written originally in [Isotonic.jl](https://github.com/ajtulloch/Isotonic.jl) and [MultipleTesting.jl](https://github.com/juliangehring/MultipleTesting.jl).

# References

[1] Guntuboyina, Adityanand, and Bodhisattva Sen. "Nonparametric shape-restricted regression." Statistical Science 33.4 (2018): 568-594.
