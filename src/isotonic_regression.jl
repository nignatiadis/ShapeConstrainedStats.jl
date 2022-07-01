"""
    IsotonicRegression(x[, y])

Compute the Bar index between `x` and `y`. If `y` is missing, compute
the Bar index between all pairs of columns of `x`.

"""
struct IsotonicRegression{S<:Number, T<:Number,
                          TV<:AbstractVector{T},
                          SV<:AbstractVector{S},
                          SW<:AbstractVector,
                          R} <: RegressionModel
    sorted_xs::TV
    sorted_ys::SV #sorted by xs
    sorted_ws::SW
    fitted_ys::SV
    rank_xs::R
    lb::S
    ub::S
end

function fit(::Type{IsotonicRegression}, X::AbstractVector, Y::AbstractVector;
             wts=Ones(Y),
             lb=nothing, ub=nothing)

    n = length(X)
    if n != length(Y)
        throw(DimensionMismatch("Lengths of X and Y mismatch"))
    end

    rank_xs = sortperm(X)
    sorted_xs = X[rank_xs]
    inv_rank_xs = invperm(rank_xs) # X == sorted_xs[inv_rank_Xs]
    sorted_ys = Y[rank_xs]
    sorted_ws = wts[rank_xs]

    sorted_xs_unique, lens_sorted_xs = rle(sorted_xs)
    n_unique = length(sorted_xs_unique)

    if n_unique == n
        sorted_xs_unique = sorted_xs
        sorted_ys_unique = sorted_ys
        sorted_ws_unique = sorted_ws
        denserank_xs = inv_rank_xs
    else
        sorted_ys_unique = zeros(eltype(sorted_ys), n_unique)
        sorted_ws_unique = zeros(eltype(wts), n_unique)
        denserank_xs = zeros(eltype(rank_xs), n)
        counter = zero(Int)
        for i in Base.OneTo(n_unique)
            for j in Base.OneTo(lens_sorted_xs[i])
                counter+=1
                sorted_ys_unique[i] += sorted_ys[counter]
                sorted_ws_unique[i] += sorted_ws[counter]
                denserank_xs[counter] = i
            end
        end
        sorted_ys_unique ./= sorted_ws_unique
        denserank_xs = denserank_xs[inv_rank_xs]
    end

    if (lb === nothing) && (ub === nothing)
        lb, ub = extrema(sorted_ys_unique)
    end
    if lb === nothing
        lb = minimum(sorted_ys_unique)
    end
    if ub === nothing
        ub = maximum(sorted_ys_unique)
    end

    fitted_ys = isotonic_regression(sorted_ys_unique, sorted_ws_unique)

    fitted_ys = max.(min.(fitted_ys, ub), lb)
    isofit = IsotonicRegression(sorted_xs_unique,
            sorted_ys_unique,
            sorted_ws_unique,
            fitted_ys,
            denserank_xs,
            lb,
            ub)
    isofit
end

function fit(::Type{IsotonicRegression}, Y::AbstractVector;
              kwargs...)
    n = length(Y)
    fit(IsotonicRegression, 1:n, Y; kwargs...)
end

function fit(::Type{IsotonicRegression}, X::AbstractMatrix, Y::AbstractVector;
              kwargs...)
    X = vec(X)
    fit(IsotonicRegression, X, Y; kwargs...)
end

function isotonic_regression!(y::AbstractVector{T},
                              weights::AbstractVector{S}) where {T,S}
    n = length(y)
    if n <= 1
        return y
    end
    if n != length(weights)
        throw(DimensionMismatch("Lengths of values and weights mismatch"))
    end
    @inbounds begin
        n -= 1
        while true
            i = 1
            is_pooled = false
            while i <= n
                k = i
                while k <= n && y[k] >= y[k+1]
                    k += 1
                end
                if y[i] != y[k]
                    numerator = zero(T)
                    denominator = zero(T)
                    for j in i:k
                        numerator += y[j] * weights[j]
                        denominator += weights[j]
                    end
                    m = numerator / denominator
                    for j in i:k
                        y[j] = m
                    end
                    is_pooled = true
                end
                i = k + 1
            end
            if !is_pooled
               break
            end
        end
    end
    return y
end

function isotonic_regression(y, weights=Ones(y))
    isotonic_regression!(copy(y), weights)
end

length(iso::IsotonicRegression) = 1
broadcast(iso::IsotonicRegression) = Ref(iso)



function predict(iso::IsotonicRegression, x::Number)
    sorted_xs = iso.sorted_xs
    fitted_ys = iso.fitted_ys

    n = length(fitted_ys)

    idxl = searchsortedlast(iso.sorted_xs, x)
    idxr = idxl + 1

    if idxl == 0
        y_fit = fitted_ys[1]
    elseif idxl == n
        y_fit = fitted_ys[n]
    else
        y_l = fitted_ys[idxl]
        y_r = fitted_ys[idxr]
        x_l = sorted_xs[idxl]
        x_r = sorted_xs[idxr]
        λ = (x_l == x_r) ? zero(eltype(y_l)) : (x - x_l)/(x_r - x_l)
        y_fit = λ*y_r + (1-λ)*y_l
    end
    y_fit
end

# TODO: Make below efficient
predict(iso::IsotonicRegression, xs) = predict.(Ref(iso), xs)

function predict(iso::IsotonicRegression)
    iso.fitted_ys[iso.rank_xs]
end
