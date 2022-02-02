using Distances

export Normalised

struct Normalised{T<:Metric} <: Metric
    d::T
end 

function (d̄::Normalised{T})(x,y) where {T<:Metric}
    d = d̄.d
    d_tmp = d(x,y)
    return 2 * d_tmp / (d(x,nothing) + d(y,nothing) + d_tmp)
end 