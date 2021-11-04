using LinearAlgebra
export hamming_dist, jaccard_dist, diffusion_dist

# A few graph distances. Operate between binary matrices.

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