export SisPosteriorSampler, SisIexInsertDelete, SisIexInsertDeleteGibbs

abstract type SisPosteriorSampler end 

struct SisIexInsertDeleteGibbs <: SisPosteriorSampler 
    ν::Int 
    β::Float64 
    path_dist::PathDistribution
    K::Int 
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{Int} # Storage for valuse to insert in Gibbs scan
    function SisIexInsertDeleteGibbs(
        path_dist::PathDistribution;
        K=100,
        ν=4, β=0.6,
        desired_samples=1000, lag=1, burn_in=0
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν)
        ind_add = zeros(Int, ν)
        vals = zeros(Int, ν)
        par_info = Dict()
        par_info[:ν] = "(maximum number of edit operations in iMCMC-within-Gibbs conditional updates)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν, β, path_dist, K,
            desired_samples, burn_in, lag,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals
            )
    end 
end 

struct SisIexInsertDelete<: SisPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε::Float64 # Neighborhood for sampling γ
    aux_mcmc::SisMcmcSampler
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{Int} # Storage for values where are deleted from interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td_add::Vector{Int} # Storage of where to insert/delete 
    ind_td_del::Vector{Int} # Storage of where to insert/delete 
    function SisIexInsertDelete(
        aux_mcmc::SisMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.8,1,K),
        ε=0.05,
        desired_samples=1000, lag=1, burn_in=0
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td_add = zeros(Int, ν_td)
        ind_td_del = zeros(Int, ν_td)
        par_info = Dict()
        par_info[:ν_ed] = "(maximum number of edit operations)"
        par_info[:ν_td] = "(maximum change in dimension)"
        par_info[:len_dist] = "(distribution to sample length of path insertions)"
        par_info[:α] = "(controls how much data informs entry insertion proposals)"
        par_info[:aux_mcmc] = "(mcmc sampler from auxiliary data)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"
        par_info[:ε] = "(neighborhood for γ proposals)"

        new(
            ν_ed, ν_td, β, α,
            len_dist, 
            ε,
            aux_mcmc, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td_add, ind_td_del
            )
    end 
end 
