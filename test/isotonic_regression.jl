using Test
using ShapeConstrainedStats
using StatsBase

@testset "Isotonic regression helper" begin

     #pooled adjacent violators example page 10 robertson
     isotonic_regression = ShapeConstrainedStats.isotonic_regression
     @test isotonic_regression([22.5; 23.333; 20.833; 24.25], [3.0;3.0;3.0;2.0]) ≈ [22.222; 22.222; 22.222; 24.25]

     #if input already ordered, then output should be the same
     @test isotonic_regression([1.0; 2.0; 3.0]) ≈ [1.0; 2.0; 3.0]
     @test isotonic_regression([1., 41., 51., 1., 2., 5., 24.], [1., 2., 3., 4., 5., 6., 7.]) ≈ [1.0, 13.95, 13.95, 13.95, 13.95, 13.95, 24.]

     # single value or empty vector remains unchanged
     r = [rand(1)]
     @test isotonic_regression(r) == r
     r = Vector{Float64}()
     @test isotonic_regression(r) == r

     r = rand(10)
     @test isotonic_regression(r, fill(1.0, size(r))) == isotonic_regression(r)
     @test_throws DimensionMismatch isotonic_regression(rand(10), ones(5))

end

robertson_vec = [22.5; 23.333; 20.833; 24.25]
robertson_ws =  [3.0;3.0;3.0;2.0]
robertson_fit = [22.222; 22.222; 22.222; 24.25]

@testset "Isotonic regression type" begin

    @test maximum(predict(fit(IsotonicRegression, robertson_vec; lb=0.0, ub=24.0))) == 24.0

    xs = rand(4)
    rank_xs = ordinalrank(xs)

    iso_fit = fit(IsotonicRegression, xs, robertson_vec[rank_xs])

    iso_fit_xs_sorted = fit(IsotonicRegression, sort(rand(4)), robertson_vec)
    iso_fit_no_xs = fit(IsotonicRegression,  robertson_vec)
    @test predict(iso_fit_xs_sorted) == predict(iso_fit_no_xs)

    iso_fit_xs_sorted_ws = fit(IsotonicRegression, sort(rand(4)), robertson_vec; wts= robertson_ws)
    iso_fit_xs_sorted_ws_type = fit(IsotonicRegression, sort(rand(4)), robertson_vec; wts= Weights(robertson_ws))
    iso_fit_no_xs_sorted_ws_type = fit(IsotonicRegression, robertson_vec; wts= Weights(robertson_ws))

    @test predict(iso_fit_xs_sorted_ws) ≈ robertson_fit

    @test predict(iso_fit_xs_sorted_ws_type) ≈ robertson_fit

    @test predict(iso_fit_no_xs_sorted_ws_type)  ≈ robertson_fit

    xs_mat = reshape(xs, 4, 1)

    @test predict(fit(IsotonicRegression, xs_mat, robertson_vec; wts= robertson_ws)) ==
         predict(fit(IsotonicRegression, xs, robertson_vec; wts= robertson_ws))

    @test predict(iso_fit_no_xs_sorted_ws_type, 1) ≈ robertson_fit[1]
    @test predict(iso_fit_no_xs_sorted_ws_type, -10) ≈ robertson_fit[1]
    @test predict(iso_fit_no_xs_sorted_ws_type, 20) ≈ robertson_fit[4]
    @test predict(iso_fit_no_xs_sorted_ws_type, 3.5) ≈ (robertson_fit[3] + robertson_fit[4])/2

    random_idx = sample([1,2,3,4],10)
    @test predict(iso_fit_no_xs_sorted_ws_type, random_idx) ≈ robertson_fit[random_idx]
end
