using Distances, LinearAlgebra, Distributions
export SNF, MultigraphSNF, SnfPosterior, MultigraphSnfPosterior

struct SNF{T<:Union{Int,Bool}}
    mode::Matrix{T}
    γ::Real
    d::Metric
    V::Int
    directed::Bool
    self_loops::Bool
end 

function SNF(
    mode::Matrix{S}, γ::Real, d::Metric;
    directed::Bool=!issymmetric(mode),        # Leave these as kw args so they can be specified if desired
    self_loops::Bool=any(diag(mode).>0)     # e.g. if you want a distribution over directed graphs where 
    ) where {S<:Union{Int,Bool}}            # the mode happens to be symmetric.
    
    return SNF(mode, γ, d, size(mode, 1), directed, self_loops)

end 

const MultigraphSNF = SNF{Int}

struct SnfPosterior{T<:Union{Int,Bool}}
    data::Vector{Matrix{T}}
    G_prior::SNF{T}
    γ_prior::UnivariateDistribution
    dist::Metric
    V::Int
    directed::Bool
    self_loops::Bool
    sample_size::Int 
    function SnfPosterior(
        data::Vector{Matrix{S}},
        G_prior::SNF{S},
        γ_prior::UnivariateDistribution
        ) where {S<:Union{Int,Bool}}
        dist, V, directed, self_loops = (
            G_prior.dist,
            G_prior.V,
            G_prior.directed, 
            G_prior.self_loops
            )
        new{S}(
            data, 
            G_prior, 
            γ_prior, 
            dist, V, directed, self_loops, length(data)
        )
    end 
end 

const MultigraphSnfPosterior = SnfPosterior{Int} 