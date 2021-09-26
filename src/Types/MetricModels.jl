using ProgressMeter, Multisets
export SPF, SIS, SIM, sum_of_dists, cardinality
export get_normalising_const, get_sample_space, eachpath, eachinterseq
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
        return V * (V^model.K - 1) / (V - 1)
    end 
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
    # Inner constructor (to impose default value for K's).
    # function SIS(
    #     mode::Vector{Path{T}},
    #     γ::Real,
    #     dist::InteractionSeqDistance,
    #     V::Vector{T};
    #     K_inner::Real,
    #     K_outer::Real
    #     ) where {T<:Union{Int, String}}
    #     new{T}(mode, γ, dist, V, K_inner, K_outer)
    # end
end

SIS(
    mode::InteractionSequence{T}, 
    γ::Real, 
    dist::InteractionSeqDistance, 
    V::Vector{T}
    ) where {T<:Union{Int,String}}= SIS(mode, γ, dist, V, Inf, Inf)

# function logpdf_propto(
#     model::Union{SIS{T}, SIM{T}},
#     x::InteractionSequence{T}
#     ) where {T<:Union{Int,String}}

#     return - model.γ * (
#         model.dist(x, model.mode)
#     )
# end 

struct SIM{T<:Union{Int, String}}
    mode::Vector{Path{T}} # Mode
    γ::Real # Precision
    dist::InteractionSetDistance # Distance metric
    V::Vector{T} # Vertex Set
    K_inner::Real # Maximum interaction sequence size
    K_outer::Real # Maximum path (interaction) length
    # Inner constructor (to impose default value for K's).
    function SIM(
        mode::Vector{Path{T}},
        γ::Real,
        dist::InteractionSetDistance,
        V::Vector{T};
        K_inner::Real=Inf,
        K_outer::Real=Inf
        ) where {T<:Union{Int, String}}
        new{T}(mode, γ, dist, V, K_inner, K_outer)
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