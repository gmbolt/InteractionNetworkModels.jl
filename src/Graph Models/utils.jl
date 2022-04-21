using LinearAlgebra
export sample_urn_without_replacement, adj_mat_to_vec, vec_to_adj_mat
export rand_perturb, get_num_vertices

function sample_urn_without_replacement(
    n::Int, bins::Vector{Int}
    )
    var_bins = copy(bins)
    n_tot = sum(bins)
    @assert n_tot ≥ n "Sum of bins must be ≥ n."
    out = Int[]
    for i in 1:n 
        counter = 0
        z = rand(1:n_tot)
        for (obs,bin_size) in enumerate(var_bins)
            counter += bin_size
            if counter ≥ z 
                var_bins[obs] -= 1
                n_tot -= 1
                push!(out, obs)
                break
            end 
        end 
    end 
    return out
end 

function adj_mat_to_vec(
        A::Matrix;
        directed::Bool=true,
        self_loops::Bool=true
    )

    V = size(A,1)
    if directed & self_loops
        return A[:]
    elseif directed & !self_loops
        return A[[i!=j ? true : false for i=1:V,j=1:V]]
    elseif !directed & self_loops
        return A[[i ≤ j ? true : false for i=1:V,j=1:V]]
    elseif !directed & !self_loops
        return A[[i < j ? true : false for i=1:V,j=1:V]]
    end
end 

function vec_to_adj_mat(
    x::Vector;
    directed::Bool=true,
    self_loops::Bool=true
    )
    M = length(x)
    if directed & self_loops
        V = round(Int, √(M))
        A = zeros(eltype(x), V, V)
        A[:] = copy(x)
        return A
    elseif directed & !self_loops
        V = round(Int, √(M+1/4) + 1/2)
        A = zeros(eltype(x), V, V)
        A[[i!=j ? true : false for i=1:V,j=1:V]] = copy(x)
        return A
    elseif !directed & self_loops
        V = round(Int, √(2M+1/4) - 1/2)
        A = zeros(eltype(x), V, V)
        A[[i ≤ j ? true : false for i=1:V,j=1:V]] = copy(x)
        A[[i > j ? true : false for i=1:V,j=1:V]] = A'[[i > j ? true : false for i=1:V,j=1:V]]
        @assert issymmetric(A) "Output must be symmetric for un-directed case."
        return A
    elseif !directed & !self_loops
        V = round(Int, √(2M+1/4) + 1/2)
        A = zeros(eltype(x), V, V)
        A[[i < j ? true : false for i=1:V,j=1:V]] = copy(x)
        A += A'
        @assert issymmetric(A) "Output must be symmetric for un-directed case."
        return A
    end 

end 

rand_perturb(A::Matrix{Bool}, τ::Float64) = [rand() < τ ? !x : x for x in A]

function get_num_vertices(
    M::Int, # Number of edges
    directed::Bool, self_loops::Bool
    )
    if directed & self_loops
        return round(Int, √(M))
    elseif directed & !self_loops
        return round(Int, √(M+1/4) + 1/2)
    elseif !directed & self_loops
        return round(Int, √(2M+1/4) - 1/2)       
    elseif !directed & !self_loops
        return round(Int, √(2M+1/4) + 1/2)
    end 
end 
