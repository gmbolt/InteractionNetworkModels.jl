using Distances, LinearAlgebra, Distributions
export SNF, SnfPosterior, MultigraphSnfPosterior
export MultigraphSNF, BinarySNF, VecBinarySNF, VecMultigraphSNF
export MultigraphSnfPosterior, BinarySnfPosterior, VecMultigraphSnfPosterior, VecBinarySnfPosterior
export vectorise

struct SNF{T<:Union{Int,Bool},N,S<:SemiMetric}
    mode::Array{T,N}
    γ::Real
    d::S
    directed::Bool
    self_loops::Bool
end 

const VecBinarySNF{S} = SNF{Bool,1,S} 
const VecMultigraphSNF{S} = SNF{Int,1,S}

const MultigraphSNF{S} = SNF{Int,2,S} 
const BinarySNF{S} = SNF{Bool,2,S}

function SNF(
    mode::Array{S,2}, γ::Real, d::SemiMetric;
    directed::Bool=!issymmetric(mode),        # Leave these as kw args so they can be specified if desired
    self_loops::Bool=any(diag(mode).>0)     # e.g. if you want a distribution over directed graphs where 
    ) where {S<:Union{Int,Bool}}            # the mode happens to be symmetric.

    return SNF(
        mode, 
        γ, d, 
        directed, self_loops
    )

end 

function SNF(
    mode::Array{S,1}, γ::Real, d::SemiMetric;
    directed::Bool=true,        # Leave these as kw args so they can be specified if desired
    self_loops::Bool=true     # e.g. if you want a distribution over directed graphs where 
    ) where {S<:Union{Int,Bool}}            # the mode happens to be symmetric.

    return SNF(
        mode, 
        γ, d, 
        directed, self_loops
    )

end 

function Base.similar(
    model::SNF{T,N,S},
    mode::Array{T,N},
    γ::Float64
    ) where {T,N,S}
    return SNF(mode, γ, model.d, model.directed, model.self_loops)
end 

Base.show(io::IO, x::SNF{T,2,S}) where {S,T} = print(io, "$(typeof(x))(γ=$(x.γ), V=$(size(x.mode,1)), directed=$(x.directed), self_loops=$(x.self_loops))")
Base.show(io::IO, x::SNF{T,1,S}) where {S,T} = print(io, "$(typeof(x))(γ=$(x.γ), E=$(length(x.mode)), directed=$(x.directed), self_loops=$(x.self_loops))")


function vectorise(model::SNF{T,2,S}) where {S,T}
    dir, sl = (model.directed, model.self_loops)
    mode_vec = adj_mat_to_vec(
        model.mode, 
        directed=dir, 
        self_loops=sl
    )
    return SNF(mode_vec, model.γ, model.d, directed=dir, self_loops=sl)
end 

struct SnfPosterior{T<:Union{Int,Bool},N,V<:SemiMetric,S<:UnivariateDistribution}
    data::Vector{Array{T,N}}
    G_prior::SNF{T,N,V}
    γ_prior::S
    d::V
    directed::Bool
    self_loops::Bool
    sample_size::Int 
    function SnfPosterior(
        data::Vector{Array{S,N}},
        G_prior::SNF{S,N,V},
        γ_prior::UnivariateDistribution
        ) where {S<:Union{Int,Bool},N,V<:SemiMetric}

        d, directed, self_loops = (
            G_prior.d,
            G_prior.directed, 
            G_prior.self_loops
        )
        new{S,N,V,typeof(γ_prior)}(
            data, 
            G_prior, 
            γ_prior, 
            d, directed, self_loops, length(data)
        )
    end 
end 

# We want the prior to drive how the posterior is structured, e.g.
# if the prior if vectorised, then so is the posterior. However,
# one might still want to pass data as matrices. Thus the data must be
# pre-processed slighlty before constructing posterior. 

function SnfPosterior(
    data::Vector{Array{S,2}},
    G_prior::SNF{S,1,V},
    γ_prior::UnivariateDistribution
    ) where {S<:Union{Int,Bool},V<:SemiMetric}

    dir, sl = (G_prior.directed, G_prior.self_loops)
    data_vec =  adj_mat_to_vec.(data, directed=dir, self_loops=sl)
    return SnfPosterior(
        data_vec, 
        G_prior, 
        γ_prior
    )
end 

function SnfPosterior(
    data_vec::Vector{Array{S,1}},
    G_prior::SNF{S,2,V},
    γ_prior::UnivariateDistribution
    ) where {S<:Union{Int,Bool},V<:SemiMetric}

    dir, sl = (G_prior.directed, G_prior.self_loops)
    data =  vec_to_adj_mat.(data_vec, directed=dir, self_loops=sl)
    return SnfPosterior(
        data, 
        G_prior, 
        γ_prior
    )
end 

const MultigraphSnfPosterior{S,T} = SnfPosterior{Int,2,V,S} where {V<:SemiMetric,S<:UnivariateDistribution} 
const BinarySnfPosterior{S,T} = SnfPosterior{Bool,2,V,S} where {V<:SemiMetric,S<:UnivariateDistribution} 

const VecMultigraphSnfPosterior{S,T} = SnfPosterior{Int,1,V,S} where {V<:SemiMetric,S<:UnivariateDistribution} 
const VecBinarySnfPosterior{S,T} = SnfPosterior{Bool,1,V,S} where {V<:SemiMetric,S<:UnivariateDistribution} 

function vectorise(posterior::SnfPosterior{T,2,S}) where {S,T}
    dir, sl = (posterior.directed, posterior.self_loops)
    data_vec =  [adj_mat_to_vec(x, directed=dir, self_loops=sl) for x in posterior.data]
    G_prior_vec = vectorise(posterior.G_prior)
    return SnfPosterior(
        data_vec, 
        G_prior_vec,
        posterior.γ_prior
    )
end 

# function SnfPosterior(
#     data::Vector{Vector{S}},
#     G_prior::SNF{S},
#     γ_prior::UnivariateDistribution
#     ) where {S<:Union{Int,Bool}}
#     directed, self_loops = (
#         G_prior.directed, 
#         G_prior.self_loops
#     )
#     data_mat = [vec_to_adj_mat(x, directed=directed, self_loops=self_loops) for x in data]
#     SnfPosterior(
#         data_mat, data, 
#         G_prior, γ_prior
#     )

# end     

# function SnfPosterior(
#     data::Vector{Matrix{S}},
#     G_prior::SNF{S},
#     γ_prior::UnivariateDistribution
#     ) where {S<:Union{Int,Bool}}
#     directed, self_loops = (
#         G_prior.directed, 
#         G_prior.self_loops
#     )
#     data_vec = [adj_mat_to_vec(x, directed=directed, self_loops=self_loops) for x in data]
#     SnfPosterior(
#         data, data_vec,
#         G_prior, γ_prior
#     )
# end     

