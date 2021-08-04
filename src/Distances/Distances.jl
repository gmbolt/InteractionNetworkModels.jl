using LinearAlgebra, StatsBase, OptimalTransport, Distances
export hamming_dist, jaccard_dist, diffusion_dist, lcs_dist, lcs_dist_normed, emd_dist

"""
Compute the Hamming distance between two adjacency matrices
"""
function hamming_dist(A1::Array{Int,2}, A2::Array{Int,2})
    return sum(map(abs, A1 - A2)) / 2
end
"""
Compute the Jaccard distance between two adjacency matrices
"""
function jaccard_dist(A1::Array{Int,2}, A2::Array{Int,2})
    unique_edges = sum((A1 + A2) .> 0) / 2 # Number of unique edges (ie union cardinality)
    return hamming_dist(A1, A2) / unique_edges
end
"""
Compute the diffusion distance between two adjacency matrices with t the time
of diffusion.
"""
function diffusion_dist(A1::Array{Int,2}, A2::Array{Int,2}, t)
    L1 = Diagonal(vec(sum(A1, dims = 2))) - A1
    L2 = Diagonal(vec(sum(A2, dims = 2))) - A2
    function matrixExp(A, t)
        F = eigen(A)
        return F.vectors * Diagonal(exp.(F.values * t)) * transpose(F.vectors)
    end
    return sum((matrixExp(-L1, t) - matrixExp(-L2, t)) .^ 2)
end
"""
Compute LCS distance
"""
function lcs_dist(X::Array{String,1}, Y::Array{String,1})::Float32

    n = length(X)
    m = length(Y)

    # If one or more is empty string return 0 or length of nonzero
    if (n == 0) || (m == 0)
        return n + m
    end

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2
    prevRow = 0:m
    currRow = zeros(Int, m + 1)
    for i = 1:n
        currRow[1] = i
        for j = 1:m
            if X[i] == Y[j]
                currRow[j+1] = prevRow[j]
            else
                currRow[j+1] = min(currRow[j], prevRow[j+1]) + 1
            end
        end
        prevRow = copy(currRow)
    end
    return currRow[end]
end
"""
Computed normalised LCS distance
"""
function lcs_dist_normed(X::Array{String,1}, Y::Array{String,1})::Float32
    n = length(X)
    m = length(Y)
    if (n == 0) & (m == 0)
        return 0
    end
    d_lcs = lcs_dist(X, Y)
    return 2 * d_lcs / (n + m + d_lcs)
end


## Defining Pairwise for Mutlisets/Vectors of any type

# Multisets
# in place
"""
In-place distance matrix calculation between elements of multiset.
"""
function Distances.pairwise!(
    A::AbstractMatrix,
    metric::Metric,
    a::Multiset{T} where T,
    b::Multiset{T} where T
    )

    for (j, val_b) in enumerate(b)
        for (i, val_a) in enumerate(a)
            A[i,j] = metric(val_a,val_b)
        end
    end
end
"""
Distance matrix calculation between elements of multiset. This is a custom extension
of the function in the Distances.jl package to allow vectors of general type.
"""
function Distances.pairwise(
    metric::Metric,
    a::Multiset{T} where T,
    b::Multiset{T} where T
    )

    D = Array{Float64,2}(undef, length(a), length(b))
    pairwise!(D, metric, a, b)
    return D
end

# Vectors
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
Distance matrix calculation between elements of Vectors. This is a custom extension
of the function in the Distances.jl package to allow vectors of general type. The
function in Distances.jl is designed for univariate/multivariate data and so takes
as input either vectors or matrices (data points as rows).
"""
function Distances.pairwise(
    metric::Metric,
    a::Vector{T} where T,
    b::Vector{T} where T
    )

    D = Array{Float64,2}(undef, length(a), length(b))
    for j in 1:length(b)
        for i in 1:length(a)
            D[i,j] = metric(a[i],b[j])
        end
    end
    return D
end
