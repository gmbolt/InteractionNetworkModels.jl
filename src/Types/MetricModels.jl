using ProgressMeter, Multisets, StatsBase, ProgressMeter
export SPF, SIS, SIM, sum_of_dists, cardinality
export get_normalising_const, get_sample_space, eachpath, eachinterseq
export get_entropy, get_entropy_nieve, get_dist_counts, get_ball_counts, get_uniform_avg_dist
export pmf_unormalised, get_true_dist_vec, get_true_dist_dict


struct SPF{T<:Union{Int, String}}
    mode::Path{T} # Mode
    γ::Real # Precision
    dist::PathDistance # Distance metric
    V::Vector{T} # Vertex Set
    K::Real # Maximum length path
end

# Outer constructors
SPF(Iᵐ, γ, d, 𝒱) = SPF(Iᵐ, γ, d, 𝒱, Inf)  # K defaults to ∞
SPF(Iᵐ, γ, d) = SPF(Iᵐ, γ, d, unique(Iᵐ), Inf)  # Vertex set dedaults to unique vals in mods


# sum of distances to mode
"""
Calculate sufficient statistic, that is, the sum of distances to
the mode. This function takes a variable number of paths as input.
"""
function sum_of_dists(model::SPF, x::Path...)
    z = 0.0
    for P in x
        z += model.dist(P, model.mode)
    end
    return z
end
# Note, I opted to define for variable number of inputs
# then define for vector by unpacking the vector. Testing showed
# little difference to this approach and writing unique function,
# that is, looping over the vector.
"""
Given `model::SPF`, calculate sum of distances to the mode. This function takes a vector of paths as input.
"""
sum_of_dists(model::SPF, x::Vector{Path}) = sum_of_dists(model, x...)

"""
`sum_of_dists(data::Vector{Path}, ref::Path, d::PathDistance)`

Model free sum of distances to a reference. Returns the sum of distances of each `Path` in `data::Vector{Path}` to a single reference `ref::Path`, where the distance used is `d::PathDistance`.
"""
function sum_of_dists(data::Vector{Path{T}}, ref::Path{T}, d::PathDistance) where {T<:Union{Int, String}}
    z = 0.0
    for P in data
        z += d(P, ref)
    end
    return z
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

function Base.iterate(model::SPF{Int})
    return [1,1], [1,1]
end 

function Base.iterate(model::SPF{Int}, state::Vector{Int})


    next = copy(state)
    # if state[end]==model.V
    V = length(model.V)
    ind = findlast(!isequal(V), state)
    if isnothing(ind)
        is_reset = false 
    else 
        ind += 1
        is_reset = mapreduce(x->isequal(x,V), * , view(state, ind:length(state)))
    end 
    println(ind, is_reset)
    if is_reset
        num_max = mapreduce(x->isequal(x,V), + , state)
        if num_max == model.K
            return nothing 
        elseif num_max == length(state)
            for i in 1:length(next)
                next[i] = 1
            end 
            push!(next, 1)
        else 
            for i in ind:length(state)
                next[i] = 1
            end 
            next[ind-1] += 1
        end 
    else 
        next[end] += 1
    end 

    return next, next 

end

function eachpath(V,K::Int)
    return Base.Iterators.flatten(
    [Base.Iterators.product([V for j=1:k]...) for k=1:K]
    )
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
        for P in Base.Iterators.product([model.V for j=1:i]...)
            # println(Path(P...))
            Z += exp( - model.γ * model.dist([P...], model.mode) )
        end
    end
    return Z
end

"""
Returns vector with all elements in the sample space.
"""
function get_sample_space(
    model::SPF
)
    z = Vector{Path}()
    for P in eachpath(model.V, model.K)
        push!(z, [P...])
    end
    return z

end

function get_entropy_nieve(
    model::SPF;
    show_progress::Bool=true
    )

    ss = get_sample_space(model)

    f(x) = exp( - model.γ * model.dist(x, model.mode)) # (Un-normlised likelihood)
    probs = map(f, ss)
    probs /= sum(probs) # Normalise
    
    return StatsBase.entropy(probs)
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
        d_tmp = d([P...], model.mode)
        Z += exp(-γ * d_tmp)
        H += - model.γ * d_tmp * exp(-γ * d_tmp)
        if show_progress
            next!(iter)
        end 
    end 
    return log(Z) - H/Z 

end

function get_dist_counts(
    model::SPF
    )

    dists = Int.(map(x -> model.dist(x, model.mode), get_sample_space(model)))
    return countmap(dists)

end 

function get_ball_counts(
    model::SPF
    )

    dists = Int.(map(x -> model.dist(x, model.mode), get_sample_space(model)))
    dist_counts = counts(dists, 1:maximum(dists))
    return cumsum(dist_counts)
end 

function get_uniform_avg_dist(
    model::SPF
    )
    dists = map(x -> model.dist(x, model.mode), get_sample_space(model))
    return mean(dists)
end 

# ===============================================================
# Interaction Sequences/Sets
# ===============================================================

# The model Types

struct SIS{T<:Union{Int, String}}
    mode::Vector{Path{T}} # Mode
    γ::Real # Precision
    dist::InteractionSeqDistance # Distance metric
    V::Vector{T} # Vertex Set
    K_inner::Real # Maximum interaction sequence size
    K_outer::Real # Maximum path (interaction) length
end

SIS(
    mode::InteractionSequence{T}, 
    γ::Real, 
    dist::InteractionSeqDistance, 
    V::Vector{T}
    ) where {T<:Union{Int,String}}= SIS(mode, γ, dist, V, Inf, Inf)


function Base.show(
    io::IO, model::SIS{T}
    ) where {T<:Union{Int,String}}

    title = "SIS Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(model))
        println(io, par, " = $(getfield(model, par))")
    end 

end 

struct SIM{T<:Union{Int, String}}
    mode::Vector{Path{T}} # Mode
    γ::Real # Precision
    dist::InteractionSetDistance # Distance metric
    V::Vector{T} # Vertex Set
    K_inner::Real # Maximum interaction sequence size
    K_outer::Real # Maximum path (interaction) length
end

SIM(
    mode::InteractionSequence{T}, 
    γ::Real, 
    dist::InteractionSeqDistance, 
    V::Vector{T}
    ) where {T<:Union{Int,String}}= SIM(mode, γ, dist, V, Inf, Inf)

function Base.show(
    io::IO, model::SIM{T}
    ) where {T<:Union{Int,String}}

    title = "SIM Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(model))
        println(io, par, " = $(getfield(model, par))")
    end 

end 

# Sufficent staistics
"""
Given `model::Union{SIS, SIM}`, calculate the sum of distances to
the mode. This function takes a variable number of Vector{Path} as input.
"""
function sum_of_dists(model::Union{SIS{T}, SIM{T}}, x::Vector{Path{T}}...) where {T<:Union{Int, String}}
    z = 0.0
    for P in x
        z += model.dist(P, model.mode)
    end
    return z
end
"""
Given `model::Union{SIS, SIM}`, calculate the sum of distances to
the mode. This function takes a Vector{Path} as input.
"""
sum_of_dists(model::Union{SIS{T}, SIM{T}}, x::Vector{Vector{Path{T}}}) where {T<:Union{Int, String}} = sum_of_dists(model, x...)


"""
`sum_of_dists(data::Vector{Vector{Path}}, ref::Vector{Path}, d::InteractionSeqDistance)`

Model free sum of distances to a reference. Returns the sum of distances of each `Path` in `data::Vector{Path}` to a single reference `ref::Path`, where the distance used is `d::PathDistance`.
"""
function sum_of_dists(data::Vector{Vector{Path{T}}}, ref::Vector{Path{T}}, d::Union{InteractionSeqDistance, InteractionSetDistance}) where {T<:Union{Int, String}}
    z = 0.0
    for S in data
        z += d(S, ref)
    end
    return z
end 

# Evaluation of pmf
"""
Evaluate the unormalised probability of an interaction seq `x`
"""
function pmf_unormalised(
    model::Union{SIS{T}, SIM{T}}, 
    x::Vector{Path{T}}) where {T<:Union{Int, String}}

    return exp(- model.γ * model.dist(x, model.mode))
end 


# Iterating and enumerating the sample spaces
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



"""
Returns vector with all elements in the sample space.
"""
function get_sample_space(model::SIS{T}) where T <:Union{Int, String}
    z = Vector{Vector{Path{T}}}()
    for I in eachinterseq(model.V, model.K_inner, model.K_outer)
        push!(z, [Path(p...) for p in I])
    end
    return z
end


function get_sample_space(model::SIM{T}) where T <:Union{Int, String}

    z = Vector{Vector{Path{T}}}()
    for I in eachinterseq(model.V, model.K_inner, model.K_outer)
        push!(z, [Path(p...) for p in I])
    end

    z = Multiset.(z)
    return unique(z)
end 



function get_true_dist_vec(model::SIS{T}; show_progress=true) where T <:Union{Int, String}
    if show_progress
        x = Vector{Float64}()
        iter = Progress(cardinality(model), 1)  # minimum update interval: 1 second
        Z = 0.0 # Normalising constant
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = pmf_unormalised(model, [Path(p...) for p in I])  # evaluate unormalised probability
            Z += val
            push!(x, val)
            next!(iter)
        end
    else 
        x = Vector{Float64}()
        Z = 0.0 # Normalising constant
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = pmf_unormalised(model, [Path(p...) for p in I])  # evaluate unormalised probability
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
    d = Dict{Vector{Path}, Float64}()
    Z = 0.0 # Normalising constant
    prob_val = 0.0
    if show_progress
        iter = Progress(cardinality(model), 1)  # minimum update interval: 1 second
        # val = Vector{Path}()
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = [Path(p...) for p in I]
            # @show val
            prob_val = pmf_unormalised(model, val)  # evaluate unormalised probability
            Z += prob_val
            d[val] = prob_val
            next!(iter)
        end
    else 
        for I in eachinterseq(model.V, model.K_inner, model.K_outer)
            val = [Path(p...) for p in I]
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
    sample_space::Vector{Multiset{Path}}=get_sample_space(model)
)
    d = Dict{Multiset{Path}, Float64}()
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
Calculate the normalising constant of SIS
"""
function get_normalising_const(
    model::SIS
)

iter = Progress(cardinality(model), 1, "Evaluating normalising constant...")  # Loading bar. Minimum update interval: 1 second
    Z = 0.0
    for S in eachinterseq(model.V, model.K_inner, model.K_outer)
        Z += pmf_unormalised(model, [Path(p...) for p in S])
        next!(iter)
    end 
    return Z
end 