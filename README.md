# ShapeConstrainedStats.jl

[![Build Status](https://travis-ci.com/nignatiadis/ShapeConstrainedStats.jl.svg?branch=master)](https://travis-ci.com/nignatiadis/ShapeConstrainedStats.jl)
[![Coverage Status](https://coveralls.io/repos/github/nignatiadis/ShapeConstrainedStats.jl/badge.svg?branch=master)](https://coveralls.io/github/nignatiadis/ShapeConstrainedStats.jl?branch=master)


A Julia package for nonparametric shape-restricted statistics, see e.g. [1] for a recent review. Currently only implements the `IsotonicRegression` type as a `StatsBase.RegressionModel`. The main fitting function was written originally in [Isotonic.jl](https://github.com/ajtulloch/Isotonic.jl) and [MultipleTesting.jl](https://github.com/juliangehring/MultipleTesting.jl).

# References

[1] Guntuboyina, Adityanand, and Bodhisattva Sen. "Nonparametric shape-restricted regression." Statistical Science 33.4 (2018): 568-594.
