export SisIexInsertDeleteEdit, SisIexInsertDeleteGibbs, SisPosteriorSampler
export SimIexInsertDeleteEdit, SimPosteriorSampler

# SIS Model 
# ---------

abstract type SisPosteriorSampler end 

struct SisIexInsertDeleteGibbs{T<:Union{Int,String}} <: SisPosteriorSampler 
    ν::Int 
    β::Float64 
    path_dist::PathDistribution{T}
    K::Int 
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{T} # Storage for valuse to insert in Gibbs scan
    function SisIexInsertDeleteGibbs(
        path_dist::PathDistribution{S};
        K=100,
        ν=4, β=0.6,
        desired_samples=1000, lag=1, burn_in=0
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν)
        ind_add = zeros(Int, ν)
        vals = zeros(Int, ν)
        par_info = Dict()
        par_info[:ν] = "(maximum number of edit operations in iMCMC-within-Gibbs conditional updates)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν, β, path_dist, K,
            desired_samples, burn_in, lag,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals
            )
    end 
end 

struct SisIexInsertDeleteEdit{T<:Union{Int,String}} <: SisPosteriorSampler
    ν_edit::Int  # Maximum number of edit operations
    ν_trans_dim::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    ε::Float64 # Neighborhood for sampling γ
    aux_mcmc::SisMcmcSampler
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{T} # Storage for values where are deleted from interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_trans_dim::Vector{Int} # Storage of where to insert/delete 
    function SisIexInsertDeleteEdit(
        path_dist::PathDistribution{S},
        aux_mcmc::SisMcmcSampler;
        K=100,
        ν_edit=2, ν_trans_dim=1 , β=0.7, α=0.0,
        ε=0.05,
        desired_samples=1000, lag=1, burn_in=0
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν_edit)
        ind_add = zeros(Int, ν_edit)
        vals = zeros(Int, ν_edit)
        ind_update = zeros(Int, ν_edit)
        ind_trans_dim = zeros(Int, ν_trans_dim)
        par_info = Dict()
        par_info[:ν_edit] = "(maximum number of edit operations)"
        par_info[:ν_trans_dim] = "(maximum change in dimension)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:α] = "(controls how much data informs entry insertion proposals)"
        par_info[:aux_mcmc] = "(mcmc sampler from auxiliary data)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"
        par_info[:ε] = "(neighborhood for γ proposals)"

        new{S}(
            ν_edit, ν_trans_dim, β, α,
            path_dist, 
            ε,
            aux_mcmc, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_trans_dim
            )
    end 
end 

# SIM Model 
# ---------

abstract type SimPosteriorSampler end 

struct SimIexInsertDeleteEdit{T<:Union{Int,String}} <: SimPosteriorSampler
    ν_edit::Int  # Maximum number of edit operations
    ν_trans_dim::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    ε::Float64
    aux_mcmc::SimMcmcSampler
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{T} # Storage for values where are deleted from interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_trans_dim::Vector{Int} # Storage of where to insert/delete 
    function SimIexInsertDeleteEdit(
        path_dist::PathDistribution{S},
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_edit=2, ν_trans_dim=1 , β=0.7, α=0.0,
        ε=0.1,
        desired_samples=1000, lag=1, burn_in=0
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν_edit)
        ind_add = zeros(Int, ν_edit)
        vals = zeros(Int, ν_edit)
        ind_update = zeros(Int, ν_edit)
        ind_trans_dim = zeros(Int, ν_trans_dim)
        par_info = Dict()
        par_info[:ν_edit] = "(maximum number of edit operations)"
        par_info[:ν_trans_dim] = "(maximum change in dimension)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:α] = "(controls how much data informs entry insertion proposals)"
        par_info[:aux_mcmc] = "(mcmc sampler from auxiliary data)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"
        par_info[:ε] = "(neighborhood for γ proposals)"

        new{S}(
            ν_edit, ν_trans_dim, β, α,
            path_dist, 
            ε,
            aux_mcmc, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_trans_dim
            )
    end 
end 
