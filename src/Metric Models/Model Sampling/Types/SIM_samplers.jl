using Distributions
export SimMcmcSampler
export SimMcmcInsertDelete, SimMcmcInsertDeleteGibbs, SimMcmcInsertDeleteSubpath, SimMcmcInsertDeleteProportional
export SimMcmcInsertDeleteLengthCentered

abstract type SimMcmcSampler end 

# Edit Allocation Sampler 
# -----------------------

struct SimMcmcInsertDelete <: SimMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
    len_dist::DiscreteUnivariateDistribution
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
    ind_td_add::Vector{Int} # Storage of where to insert/delete 
    ind_td_del::Vector{Int} # Storage of where to insert/delete 
    function SimMcmcInsertDelete(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.4, len_dist=TrGeometric(0.8,1,K),
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
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
        par_info[:ν_td] = "(maximum increase/decrease in dimension)"
        par_info[:len_dist] = "(distribution to sample length of path insertions)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν_ed, ν_td, β, 
            len_dist, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td_add, ind_td_del
            )
    end  
end 

function Base.show(io::IO, sampler::SimMcmcInsertDelete)
    title = "MCMC Sampler for SIM Models"
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

# Edit allocation sampler with edits proportional to path size

struct SimMcmcInsertDeleteProportional <: SimMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
    len_dist::DiscreteUnivariateDistribution
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
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SimMcmcInsertDeleteProportional(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.4, len_dist=TrGeometric(0.8,1,K),
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td = zeros(Int, ν_td)
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
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td
            )
    end  
end 

function Base.show(io::IO, sampler::SimMcmcInsertDeleteProportional)
    title = "MCMC Sampler for SIM Models"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Description: interaction insertion/deletion with edit allocation updates proportional to path size.")
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

struct SimMcmcInsertDeleteLengthCentered <: SimMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
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
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SimMcmcInsertDeleteLengthCentered(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.5,
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td = zeros(Int, ν_td)
        par_info = Dict()
        par_info[:ν_ed] = "(maximum number of edit operations)"
        par_info[:ν_td] = "(maximum increase/decrease in dimension)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν_ed, ν_td, β, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td
            )
    end  
end 

function Base.show(io::IO, sampler::SimMcmcInsertDeleteLengthCentered)
    title = "MCMC Sampler for SIM Models"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Description: interaction insertion/deletion with edit allocation updates proportional to path size.")
    println(io, "Parameters:")
    num_of_pars = 4
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+4)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


# Edit Allocation Sampler (for LSP ground dist) 
# ---------------------------------------------


struct SimMcmcInsertDeleteSubpath <: SimMcmcSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
    len_dist::DiscreteUnivariateDistribution
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
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SimMcmcInsertDeleteSubpath(
        ;
        K=100,
        ν_ed=2, ν_td=2, β=0.4, len_dist=TrGeometric(0.8,1,K),
        desired_samples=1000, lag=1, burn_in=0, init=InitMode()
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td = zeros(Int, ν_td)
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
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td
            )
    end  
end 

function Base.show(io::IO, sampler::SimMcmcInsertDeleteSubpath)
    title = "MCMC Sampler for SIM Models"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Description: interaction insertion/deletion with edit allocation updates preserving subpaths.")
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



# Gibbs Scan Sampler
# ------------------

struct SimMcmcInsertDeleteGibbs<: SimMcmcSampler
    ν::Int   # Maximum number of edit ops
    path_dist::PathDistribution  # Distribution used to introduce new interactions
    β::Real  # Extra probability of Gibbs move
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{Int} # Storage for valuse to insert in Gibbs scan
    function SimMcmcInsertDeleteGibbs(
        path_dist::PathDistribution;
        K=100,
        ν=4, β=0.0,
        desired_samples=1000, lag=1, burn_in=0
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        ind_del = zeros(Int, ν)
        ind_add = zeros(Int, ν)
        vals = zeros(Int, ν)
        par_info = Dict()
        par_info[:ν] = "(maximum number of edit operations)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(extra probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new(
            ν, path_dist, β, K,
            desired_samples, burn_in, lag, 
            par_info, 
            curr_pointers, prop_pointers, ind_del, ind_add, vals
            )
    end 
end 

function Base.show(io::IO, sampler::SimMcmcInsertDeleteGibbs)
    title = "MCMC Sampler for SIM Models via Gibbs + Insert/Delete Moves."
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 4
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 
