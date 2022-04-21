using Distances
export pmf_unormalised, cardinality, eachinterseq, eachpath
export get_sample_space, get_normalising_const 
export get_true_dist_dict, get_true_dist_vec 
export get_entropy
export log_multinomial_ratio, myaddcounts!
export sample_frechet_mean, sample_frechet_mean_mem, sample_frechet_var


function sample_frechet_mean(
    data::Vector{T}, 
    d::Metric;
    show_progress::Bool=false,
    with_memory::Bool=false
    ) where T 

    if with_memory 
        return sample_frechet_mean_mem(
            data, d, show_progress=show_progress
        )
    elseif show_progress
        iter = Progress(length(data), 1)
    end 
    z_best = Inf
    ind_best = 1
    n = length(data)
    for i in 1:n
        z = 0.0
        j = 1 + (i==1)
        while (z < z_best) & (j ≤ n)
            z += d(data[i], data[j])^2
            j += 1 + (j==(i-1))
        end 
        if z < z_best 
            z_best = copy(z)
            ind_best = i
        end 
        if show_progress
            next!(iter)
        end 
    end 
    return data[ind_best], ind_best, z_best
end 

function sample_frechet_mean_mem(
    data::InteractionSequenceSample{T}, 
    d::Metric;
    show_progress::Bool=false
    ) where T <:Union{Int, String}
    if show_progress
        iter = Progress(length(data), 1)
    end 
    data_unq = unique(data)
    weights = countmap(data)
    z_best = Inf
    ind_best_unq = 1
    for i in eachindex(data_unq)
        z = 0.0
        j = 1 + (i==1)
        S1 = data_unq[i]
        while (z < z_best) & (j ≤ length(data_unq))
            S2 = data_unq[j]
            dist = d(S1,S2)
            z += weights[S2] * dist^2
            j += 1 + (j==(i-1))
        end 
        if z < z_best 
            z_best = copy(z)
            ind_best_unq = i
        end 
        if show_progress
            next!(iter)
        end 
    end 
    ind_best = findfirst(x->x==data_unq[ind_best_unq], data)
    return data[ind_best], ind_best, z_best
end 

function sample_frechet_var(
    data::InteractionSequenceSample{T}, 
    d::Metric;
    show_progress::Bool=false,
    with_memory::Bool=false
    ) where T <:Union{Int, String}

    S, i, out = sample_frechet_mean(data, d, show_progress=show_progress, with_memory=with_memory)
    return out/length(data)
end 

function myaddcounts!(d::Dict{T}, s::T) where {T}
    d[s] = get(d, s, 0) + 1
end 

# """
# For two vectors ``X = [x_1, \dots, x_N]`` and ``Y = [y_1, \dots, y_N]`` this calculates the following term
# ```math
#     log \left( \frac{N! w^\prime_1! \cdots w^\prime_{m}!}{M! w_1!\cdots w_{n}!}\right)
# ```
# where ``w_i`` and ``w_j^\prime`` represent the counts of (respectively) ``n`` and ``m`` unique entries of ``X`` and ``Y``. 

# This comes into use when looking to sample multisets via marginalising out order information of sequences. 
# """
function log_multinomial_ratio(x::AbstractVector, y::AbstractVector)
    if length(x) > length(y)
        x = log_multinomial_ratio(y,x)
        return -x
    end 
    # Now we can assume length(x) ≤ length(y)
    z = 0.0
    dx = Dict{eltype(x), Int}()  # To store values seen in x and their counts
    dy = Dict{eltype(y), Int}()  # To store values seen in y and their counts

    # First sort the element counts terms
    @inbounds for i in eachindex(x) 
        # @show x_val, y_val, typeof(x_val)
        myaddcounts!(dx, x[i])
        myaddcounts!(dy, y[i])
        z += log(dy[y[i]]) - log(dx[x[i]]) 
    end 
    @inbounds for i=(length(x)+1):(length(y))
        myaddcounts!(dy, y[i])
        z += log(dy[y[i]]) - log(i)
    end 
    return z
end 


"""
Evaluate the unormalised probability of an interaction seq `x`
"""
function pmf_unormalised(
    model::Union{SIS, SIM}, 
    x::Vector{Path{Int}})

    return exp(- model.γ * model.dist(x, model.mode))
end 


"""
Calculate the sample space cardinality.
"""
function cardinality(
    model::SIS
)::Int
    if (model.K_inner == Inf) | (model.K_outer == Inf)
        return Inf
    else 
        V = length(model.V)
        num_paths = V * (V^model.K_inner - 1) / (V - 1)
        return num_paths * (num_paths^model.K_outer - 1) / (num_paths - 1)
    end 
end 

"""
Calculate the sample space cardinality.
"""
function cardinality(
    model::SPF
)
    if model.K == Inf
        return Inf
    else 
        V = length(model.V)
        return Int(V * (V^model.K - 1) / (V - 1))
    end 
end 



"""
`eachinterseq(V, K, L)` 

Returns an iterator over all interaction sequences over the vertex set `V`, with dimension bounds defined by `K` and `L`, specifically 
* `V` = vertex set, must be a vector of unique strings or integers;
* `K` = max interaction length, must be an integer;
* `L` = max number of interactions, must be an integer.
"""
function eachinterseq(
    V::Vector{T}, 
    K_inner::Int, 
    K_outer::Int) where T <:Union{Int, String}

    return Base.Iterators.flatten(
        [Base.Iterators.product([eachpath(V, K_inner) for j = 1:k]...) for k=1:K_outer ]
    )
end 

function eachpath(V,K::Int)
    return Base.Iterators.flatten(
    [Base.Iterators.product([V for j=1:k]...) for k=1:K]
    )
end 

"""
Returns vector with all elements in the sample space.
"""
function get_sample_space(model::SIS) 
    z = Vector{Vector{Path{Int}}}()
    for I in eachinterseq(model.V, model.K_inner.u, model.K_outer.u)
        push!(z, [collect(p) for p in I])
    end
    return z
end

function get_sample_space(model::SIM)

    z = Vector{Vector{Path{Int}}}()
    for I in eachinterseq(model.V, model.K_inner.u, model.K_outer.l)
        push!(z, [collect(p) for p in I])
    end

    z = Multiset.(z)
    return unique(z)
end 


"""
Returns vector with all elements in the sample space.
"""
function get_sample_space(
    model::SPF
)
    z = Vector{Path}()
    for P in eachpath(model.V, model.K)
        push!(z, collect(P))
    end
    return z

end



"""
Calculate the normalising constant of SIS
"""
function get_normalising_const(
    model::SIS
)

iter = Progress(cardinality(model), 1, "Evaluating normalising constant...")  # Loading bar. Minimum update interval: 1 second
    Z = 0.0
    for S in eachinterseq(model.V, model.K_inner, model.K_outer)
        Z += pmf_unormalised(model, [collect(p) for p in S])
        next!(iter)
    end 
    return Z
end 

"""
Calculate the true normalising constant. 
"""
function get_normalising_const(
    model::SPF
    )

    @assert model.K < Inf "Model must be bounded (K<∞)"
    @assert typeof(model.K)==Int "K must be integer"

    Z = 0.0
    for i=1:model.K
        for P in eachpath(model.V, model.K)
            Z += exp( - model.γ * model.dist(collect(P), model.mode) )
        end
    end
    return Z
end


function get_true_dist_vec(model::SIS; show_progress=true) 
    if show_progress
        x = Vector{Float64}()
        iter = Progress(cardinality(model), 1)  # minimum update interval: 1 second
        Z = 0.0 # Normalising constant
        for I in eachinterseq(model.V, model.K_inner.u, model.K_outer.u)
            val = pmf_unormalised(model, [collect(p) for p in I])  # evaluate unormalised probability
            Z += val
            push!(x, val)
            next!(iter)
        end
    else 
        x = Vector{Float64}()
        Z = 0.0 # Normalising constant
        for I in eachinterseq(model.V, model.K_inner.u, model.K_outer.u)
            val = pmf_unormalised(model, [collect(p) for p in I])  # evaluate unormalised probability
            Z += val
            push!(x, val)
        end
    end 
    return x / Z
end

function get_true_dist_dict(
    model::SIS;
    show_progress=true
)
    d = Dict{Vector{Path{Int}}, Float64}()
    Z = 0.0 # Normalising constant
    prob_val = 0.0
    if show_progress
        iter = Progress(cardinality(model), 1)  # minimum update interval: 1 second
        # val = Vector{Path}()
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = [collect(p) for p in I]
            # @show val
            prob_val = pmf_unormalised(model, val)  # evaluate unormalised probability
            Z += prob_val
            d[val] = prob_val
            next!(iter)
        end
    else 
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = [collect(p) for p in I]
            prob_val = pmf_unormalised(model, val)  # evaluate unormalised probability
            Z += prob_val
            d[val] = prob_val
        end
    end 
    map!(x -> x/Z, values(d)) # Normalise
    return d
end


function get_true_dist_dict(
    model::SIM;
    sample_space::Vector{Multiset{Path{Int}}}=get_sample_space(model)
)
    d = Dict{Multiset{Path{Int}}, Float64}()
    Z = 0.0 # Normalising constant
    prob_val = 0.0

    # sample_space = get_sample_space(model)

    for val in sample_space
        prob_val = pmf_unormalised(model, collect(val))  # collect() turns the multiset val into a vector for passing to pmf_unormalised()
        Z += prob_val
        d[val] = prob_val
    end 
    map!(x -> x/Z, values(d)) # Normalise
    return d
end 


function get_entropy(
    model::SPF;
    show_progress::Bool=true
    )

    if show_progress 
        iter = Progress(
            cardinality(model), # How many iters 
            1,  # At which granularity to update loading bar
            "Evaluating entropy....")  # Loading bar. Minimum update interval: 1 second
    end 
    d, γ, V, K = (model.dist, model.γ, model.V, model.K) # Aliases
    Z, H = (0.0,0.0) 
    for P in eachpath(V, K)
        d_tmp = d(collect(P), model.mode)
        Z += exp(-γ * d_tmp)
        H += - model.γ * d_tmp * exp(-γ * d_tmp)
        if show_progress
            next!(iter)
        end 
    end 
    return log(Z) - H/Z 

end



