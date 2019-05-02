module ShapeConstrainedStats

    using StatsBase,
          FillArrays

    import Base.Broadcast: broadcastable
    import Base:length 

    import StatsBase:RegressionModel, fit, predict

    include("isotonic_regression.jl")

    export IsotonicRegression,
           fit,
           predict

end # module
