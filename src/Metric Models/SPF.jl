using StatsBase, Distributions, Distances
export SPF, DcSPF, SpfPosterior
export cardinality, get_normalising_const


struct SPF{T<:Union{Int, String}}
    mode::Path{T} # Mode
    γ::Real # Precision
    dist::Metric # Distance metric
    V::Vector{T} # Vertex Set
    K::Real # Maximum length path
end

SPF(Iᵐ, γ, d, 𝒱) = SPF(Iᵐ, γ, d, 𝒱, Inf)  # K defaults to ∞
SPF(Iᵐ, γ, d) = SPF(Iᵐ, γ, d, unique(Iᵐ), Inf)  # Vertex set dedaults to unique vals in mods

struct DcSPF
    mode::Path{Int} # Mode 
    γ::Real         # Dispersion 
    d::Metric
    p::DiscreteUnivariateDistribution  # Path length dist 
    V::UnitRange
end 
# K defaults to ∞
DcSPF(Iᵐ, γ, d, p) = SPF(Iᵐ, γ, d, p, unique(Iᵐ))  # Vertex set dedaults to unique vals in mods

struct SpfPosterior{T<:Union{Int, String}}
    data::Vector{Path{T}}
    I_prior::SPF{T}
    γ_prior::ContinuousUnivariateDistribution
    dist::Metric
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
