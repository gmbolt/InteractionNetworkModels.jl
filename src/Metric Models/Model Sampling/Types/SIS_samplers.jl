using Distributions
export SisMcmcSampler
export SisMcmcInsertDelete, SisMcmcInsertDeleteGibbs, SisMcmcSplitMerge
export SisMcmcInsertDeleteCenter

abstract type SisMcmcSampler end 

# From Models 

# Insert-Delete 
# -------------

struct SisMcmcInsertDeleteGibbs<: SisMcmcSampler
    ν_gibbs::Int   # Maximum number of edit ops
    ν_td::Int  # Maximum increase or decrease in number of interactions
    path_dist::PathDistribution  # Distribution used to introduce new interactions
    β::Real  # Extra probability of Gibbs move
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::McmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{Int} # Storage for valuse to insert in Gibbs scan
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SisMcmcInsertDeleteGibbs(
        path_dist::T;
        K=100,
        ν_gibbs=4, ν_td=2,  β=0.6,
        desired_samples=1000, lag=1, burn_in=0,
        init=InitMode()
        ) where {T<:PathDistribution}
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_gibbs)
        ind_add = zeros(Int, ν_gibbs)
        vals = zeros(Int, ν_gibbs)
        ind_td = zeros(Int, ν_td)
        par_info = Dict()
        par_info[:ν_gibbs] = "(maximum number of edit operations in iMCMC-within-Gibbs conditional updates)"
        par_info[:ν_td] = "(maximum number of interaction insertions or deletions)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν_gibbs, ν_td, path_dist, β, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals, ind_td
            )
    end 
end 

SisMcmcInsertDeleteGibbs(
    model::SIS;
    K=100,
    ν=4, β=0.0,
    desired_samples=1000, lag=1, burn_in=0, init=InitMode()
    ) = SisMcmcInsertDeleteGibbs(
        PathPseudoUniform(model.V, TrGeometric(0.8, 1, model.K_inner));
        K=K, ν=ν, β=β, desired_samples=desired_samples, lag=lag, burn_in=burn_in
        )


function Base.show(io::IO, sampler::SisMcmcInsertDeleteGibbs)
    title = "MCMC Sampler for SIS Models via iMCMC-within-Gibbs and Interaction Insertion/Deletion."
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 5
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+4)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


struct SisMcmcInsertDelete <: SisMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of update move
    len_dist::DiscreteUnivariateDistribution  # Dist. to sample path lengths
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::McmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td_add::Vector{Int} # Storage of where to insert/delete 
    ind_td_del::Vector{Int} # Storage of where to insert/delete 
    dist_curr::Vector{Float64}
    function SisMcmcInsertDelete(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.4, len_dist=TrGeometric(0.8,1,K),
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_update = zeros(Int, ν_ed)
        ind_td_add = zeros(Int, ν_td)
        ind_td_del = zeros(Int, ν_td)
        dist_curr = [0.0]
        par_info = Dict()
        par_info[:ν_ed] = "(maximum number of edit operations)"
        par_info[:ν_td] = "(maximum increase/decrease in dimension)"
        par_info[:len_dist] = "(distribution to sample length of path insertions)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν_ed, ν_td, β, 
            len_dist, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers,
            ind_update, ind_td_add, ind_td_del, 
            dist_curr
            )
    end 
end 

function Base.show(io::IO, sampler::SisMcmcInsertDelete)
    title = "MCMC Sampler for SIS Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Description: interaction insertion/deletion with edit allocation updates.")
    println(io, "Parameters:")
    num_of_pars = 5
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+4)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 

# Centered Insert/Delete
# ----------------------

struct SisMcmcInsertDeleteCenter <: SisMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of update move
    p::Float64  # Par for geometric on size difference
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::McmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SisMcmcInsertDeleteCenter(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.4, p=0.7,
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_update = zeros(Int, ν_ed)
        ind_td = zeros(Int, ν_td)
        par_info = Dict()
        par_info[:ν_ed] = "(maximum number of edit operations)"
        par_info[:ν_td] = "(maximum increase/decrease in dimension)"
        par_info[:p] = "(parameter for Geometric dists. on size difference from mean)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν_ed, ν_td, β, 
            p, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers,
            ind_update, ind_td
            )
    end 
end 

function Base.show(io::IO, sampler::SisMcmcInsertDeleteCenter)
    title = "MCMC Sampler for SIS Models via with Multinomial Allocated Updates and Interaction Insertion/Deletion."
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 5
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+4)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


# Merge/split
# -----------

struct SisMcmcSplitMerge <: SisMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of update move
    η::Float64  # Noisy parameter for split/merge
    p::Float64  # Parameter for Geometric dist in split/merge
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::McmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{Int} # Storage for values to insert in interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td::Vector{Int} # Storage for location of split/merges
    p_ins::Geometric 
    function SisMcmcSplitMerge(
        ;ν_ed=2, ν_td=2, β=0.7,
        η=0.7, p=0.7,
        K=100,
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td = zeros(Int, ν_td)
        p_ins = Geometric(p)
        par_info = Dict()
        par_info[:ν_ed] = "(maximum number of edit operations)"
        par_info[:ν_td] = "(maximum increase/decrease in dimension)"
        par_info[:η] = "(noise parameter for merge/splits)"
        par_info[:p] = "(parameter for Geometric dist. in merge/splits)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of paths, used to initialise storage)"

        new(
            ν_ed, ν_td, β, 
            η, p,
            K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td,
            p_ins
            )
    end 
end 

function Base.show(io::IO, sampler::SisMcmcSplitMerge)
    title = "MCMC Sampler for SIS Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Description: edit allocation with merge and split\n")
    println(io, "Parameters:")
    num_of_pars = 6
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+4)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 
