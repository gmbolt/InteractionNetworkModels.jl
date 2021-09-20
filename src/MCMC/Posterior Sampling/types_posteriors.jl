using Distributions, StatsBase

export SisPosterior, SisPosterior
export SpfPosterior, log_eval, get_length_dist

# SIS 

struct SisPosterior{T<:Union{Int, String}}
    data::Vector{Vector{Path{T}}}
    S_prior::SIS{T}
    γ_prior::ContinuousUnivariateDistribution
    dist::InteractionSeqDistance
    V::Vector{T}
    K_inner::Real
    K_outer::Real
    sample_size::Int
    function SisPosterior(
        data::Vector{Vector{Path{S}}},
        S_prior::SIS{S}, 
        γ_prior::ContinuousUnivariateDistribution
        ) where {S<:Union{Int, String}}

        dist = S_prior.dist
        V = S_prior.V
        K_inner = S_prior.K_inner
        K_outer = S_prior.K_outer
        sample_size = length(data)
        new{S}(data, S_prior, γ_prior, dist, V, K_inner, K_outer, sample_size)
    end 
end 



# SIM 

struct SimPosterior{T<:Union{Int, String}}
    data::Vector{Vector{Path{T}}}
    S_prior::SIM{T}
    γ_prior::ContinuousUnivariateDistribution
    dist::InteractionSetDistance
    V::Vector{T}
    K_inner::Real
    K_outer::Real
    sample_size::Int
    function SimPosterior(
        data::Vector{Vector{Path{S}}},
        S_prior::SIM{S}, 
        γ_prior::ContinuousUnivariateDistribution
        ) where {S<:Union{Int, String}}

        dist = S_prior.dist
        V = S_prior.V
        K_inner = S_prior.K_inner
        K_outer = S_prior.K_outer
        sample_size = length(data)
        new{S}(data, S_prior, γ_prior, dist, V, K_inner, K_outer, sample_size)
    end 
end 


# Spherical Path Family (SPF)

struct SpfPosterior{T<:Union{Int, String}}
    data::Vector{Path{T}}
    I_prior::SPF{T}
    γ_prior::ContinuousUnivariateDistribution
    dist::PathDistance
    V::Vector{T}
    K::Real
    sample_size::Int
    function SpfPosterior(
        data::Vector{Path{S}}, 
        I_prior::SPF{S}, 
        γ_prior::ContinuousUnivariateDistribution
        ) where {S<:Union{Int, String}}

        dist = I_prior.dist
        V = I_prior.V
        K = I_prior.K
        sample_size = length(data)
        new{S}(data, I_prior, γ_prior, dist, V, K, sample_size)
    end 
end 

function log_eval(p::SpfPosterior, Z::Float64, Iᵐ::Path, γ::Float64)
    log_posterior = (
        -γ * sum_of_dists(p.data, Iᵐ, p.dist) # Unormalised log likleihood
        - p.sample_size * log(Z) # Normalising constant
        - p.I_prior.γ * p.dist(Iᵐ, p.I_prior.mode) # Iᵐ prior (unormalised)
        + logpdf(p.γ_prior, γ)

    )
    return log_posterior 

end 

function get_length_dist(p::SpfPosterior{T}, α::Float64) where T <:Union{Int, String}
    
    lprobs = counts(length.(p.data), 1:p.K) .+ α * length(p.data)
    lprobs = lprobs / sum(lprobs)

    return Categorical(lprobs)
end 


function get_vertex_proposal_dist(p::SpfPosterior{T}) where {T<:Union{Int, String}}
    μ = vertex_counts(vcat(p.data...), p.V)
    μ /= sum(μ)
    return μ
end 