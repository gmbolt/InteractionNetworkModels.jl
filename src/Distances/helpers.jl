using Distances, StatsBase, ProgressMeter

# Here we simply extend the pairwise() and pairwise!() functions of the Distances.jl package. 

"""
In-place distance matrix calculation between elements of Vectors.
"""
function Distances.pairwise!(
    A::AbstractArray,
    metric::Metric,
    a::Vector{T} where T,
    b::Vector{T} where T
    )

    for j in 1:length(b)
        for i in 1:length(a)
            A[i,j] = metric(a[i],b[j])
        end
    end
end
"""
In-place distance matrix calculation between elements of Vectors. (For SubArray)
"""
function Distances.pairwise!(
    A::SubArray,
    metric::Metric,
    a::Vector{T} where T,
    b::Vector{T} where T
    )

    for j in 1:length(b)
        for i in 1:length(a)
            A[i,j] = metric(a[i],b[j])
        end
    end
end

"""
In-place distance matrix calculation between elements of Vectors. (For SubArray)
"""
function Distances.pairwise!(
    A::AbstractMatrix,
    metric::Metric,
    a::Vector{T} where T,
    b::Vector{T} where T
    )

    for j in 1:length(b)
        for i in 1:length(a)
            A[i,j] = metric(a[i],b[j])
        end
    end
end

"""
Distance matrix calculation between elements of Vectors. This is a custom extension
of the function in the Distances.jl package to allow vectors of general type. The
function in Distances.jl is designed for univariate/multivariate data and so takes
as input either vectors or matrices (data points as rows).
"""
function Distances.pairwise(
    metric::Metric,
    a::Vector{T},
    b::Vector{T} 
    ) where {T}
    D = Array{Float64,2}(undef, length(a), length(b))
    for j in 1:length(b)
        for i in 1:length(a)
            D[i,j] = metric(a[i],b[j])
        end
    end
    return D
end

"""
Distance matrix calculation between elements of Vectors. This is a custom extension
of the function in the Distances.jl package to allow vectors of interaction sequences.
"""
function Distances.pairwise(
    d::Metric,
    a::Vector{T};
    show_progress=false
    ) where {T<:Union{InteractionSequence{Int},InteractionSequence{String}}}

    D = zeros(length(a), length(a))
    if show_progress
        iter = Progress(Int(length(a)*(length(a)-1)/2), 1)
        for j in 1:length(a)
            for i in 1:(j-1)
                D[i,j] = d(a[i],a[j])
                next!(iter)
            end
        end
        D += D'
        return D
    else 
        for j in 1:length(a)
            for i in 1:(j-1)
                D[i,j] = d(a[i],a[j])
            end
        end
        D += D'
        return D
    end 
end

function StatsBase.counts(
    data::Vector{String},
    ref::Vector{String}
    )
    out = zeros(Int,length(ref))
    for v in data 
        ind = findfirst(x->x==v,ref)
        out[ind] += 1
    end 
    return out 
end