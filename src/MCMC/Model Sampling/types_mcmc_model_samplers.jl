export SisInvolutiveMcmcMergeSplit, SimInvolutiveMcmcMergeSplit
export SimInvolutiveMcmcInsertDelete, SisInvolutiveMcmcInsertDelete
export SpfMcmcSampler, SisMcmcSampler, SimMcmcSampler
export SpfInvolutiveMcmcCentSubseq, SpfInvolutiveMcmcEdit
# ====================
#   SPF 
# ====================

# From Model
abstract type SpfMcmcSampler end

struct SpfInvolutiveMcmcEdit <: SpfMcmcSampler
    ν::Int 
    desired_samples::Int
    burn_in::Int 
    lag::Int
    par_info::Dict
    curr::Vector{Int} # Storage for current val
    prop::Vector{Int} # Storage for proposed val
    ind_del::Vector{Int} # Storage for indexing of deletions
    ind_add::Vector{Int} # Storage for indexing of additions
    vals::Vector{Int} # Storgae for new values to insert
    function SpfInvolutiveMcmcEdit(
        ;ν=4, desired_samples=1000, burn_in=0, lag=1
    )
    # req_samples = burn_in + 1 + (desired_samples - 1) * lag
    curr = Int[]
    prop = Int[]
    ind_del = zeros(Int, ν)
    ind_add = zeros(Int, ν)
    vals = zeros(Int, ν)
    par_info = Dict()
    par_info[:ν] = "(maximum number of edit operations)"
    new(
        ν, 
        desired_samples, burn_in, lag, par_info, 
        curr, prop, ind_del, ind_add, vals
        )
    end 
end 

function Base.show(io::IO, sampler::SpfInvolutiveMcmcEdit)
    title = "MCMC Sampler for Spherical Path Family (SPF) via Edit Operations"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 1
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


struct SpfInvolutiveMcmcCentSubseq <: SpfMcmcSampler
    p::Real
    ν::Int
    desired_samples::Int
    burn_in::Int 
    lag::Int
    par_info::Dict
    function SpfInvolutiveMcmcCentSubseq(
        ;p=0.6, ν=4, desired_samples=1000, burn_in=0, lag=1
    )
    par_info = Dict()
    par_info[:p] = "(preserved subseq size)"
    par_info[:ν] = "(diff in path length)"
    new(p, ν, desired_samples, burn_in, lag, par_info)
    end 
end 

function Base.show(io::IO, sampler::SpfInvolutiveMcmcCentSubseq)
    title = "MCMC Sampler for Spherical Path Family (SPF) via Subsequence Preservation"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 2
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


# =====================================
#      SIS / SIM
# =====================================


abstract type SisMcmcSampler end 
abstract type SimMcmcSampler end 

# From Models 

# Insert-Delete 
# -------------

struct SisInvolutiveMcmcInsertDelete{T<:Union{Int,String}} <: SisMcmcSampler
    ν::Int   # Maximum number of edit ops
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    β::Real  # Extra probability of Gibbs move
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{T} # Storage for valuse to insert in Gibbs scan
    function SisInvolutiveMcmcInsertDelete(
        path_dist::PathDistribution{S};
        K=100,
        ν=4, β=0.0,
        desired_samples=1000, lag=1, burn_in=0
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν)
        ind_add = zeros(Int, ν)
        vals = zeros(Int, ν)
        par_info = Dict()
        par_info[:ν] = "(maximum number of edit operations)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(extra probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν, path_dist, β, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals
            )
    end 
end 

SisInvolutiveMcmcInsertDelete(
    model::SIS;
    K=100,
    ν=4, β=0.0,
    desired_samples=1000, lag=1, burn_in=0
    ) = SisInvolutiveMcmcInsertDelete(
        PathPseudoUniform(model.V, TrGeometric(0.8, 1, model.K_inner));
        K=K, ν=ν, β=β, desired_samples=desired_samples, lag=lag, burn_in=burn_in
        )


function Base.show(io::IO, sampler::SisInvolutiveMcmcInsertDelete)
    title = "MCMC Sampler for Spherical Interaction Sequence (SIS) Family via Insert-Delete Algorithm"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 3
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 

struct SimInvolutiveMcmcInsertDelete{T<:Union{Int,String}} <: SimMcmcSampler
    ν::Int   # Maximum number of edit ops
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    β::Real  # Extra probability of Gibbs move
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{T} # Storage for valuse to insert in Gibbs scan
    function SimInvolutiveMcmcInsertDelete(
        path_dist::PathDistribution{S};
        K=100,
        ν=4, β=0.0,
        desired_samples=1000, lag=1, burn_in=0
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν)
        ind_add = zeros(Int, ν)
        vals = zeros(Int, ν)
        par_info = Dict()
        par_info[:ν] = "(maximum number of edit operations)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(extra probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν, path_dist, β, K,
            desired_samples, burn_in, lag, 
            par_info, 
            curr_pointers, prop_pointers, ind_del, ind_add, vals
            )
    end 
end 

function Base.show(io::IO, sampler::SimInvolutiveMcmcInsertDelete)
    title = "MCMC Sampler for Spherical Interaction Multiset (SIM) Family via Insert-Delete Algorithm"
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

# Merge-Split (Old method)
# ------------------------

struct SisInvolutiveMcmcMergeSplit <: SisMcmcSampler
    p1::Real # Controls preserved subseq size 
    p2::Real # Controls change in path length for merge/split moves
    β::Real # Extra probability of doing a Gibbs move
    ν_outer::Int # neighbourhood for uniform sampling of change in number of interactions
    ν_kick::Int # neighbourhood for uniform sampling of kick in path lengths of merge/split moves
    ν_gibbs::Int # neighbourhood for uniform sampling of path lengths in Gibbs entry updates
    desired_samples::Int # We now have some fields which control ouput. These will be default values for resulitng sampler
    burn_in::Int
    lag::Int
    par_info::Dict # Tag line describing each parameter
    function SisInvolutiveMcmcMergeSplit(
        ;p1=0.8, p2=0.8, β=0.0, ν_outer=2, ν_kick=2, ν_gibbs=4, desired_samples=1000, burn_in=0, lag=1
        )
        par_info = Dict()
        par_info[:p1] = "(preserved subseq size)"
        par_info[:p2] = "(diff in path length on merge/split moves)"
        par_info[:β] = "(extra prob. of Gibbs move)"
        par_info[:ν_outer] = "(neighbourhood for number of paths)"
        par_info[:ν_kick] = "(neighbourhood for kick)"
        par_info[:ν_gibbs] = "(neighbourhood for path length in Gibbs move)"  
        new(p1, p2, β, ν_outer, ν_kick, ν_gibbs, desired_samples, burn_in, lag, par_info)
    end 
end 

struct SimInvolutiveMcmcMergeSplit <: SisMcmcSampler
    p1::Real # Controls preserved subseq size 
    p2::Real # Controls change in path length for merge/split moves
    β::Real # Extra probability of doing a Gibbs move
    ν_outer::Int # neighbourhood for uniform sampling of change in number of interactions
    ν_kick::Int # neighbourhood for uniform sampling of kick in path lengths of merge/split moves
    ν_gibbs::Int # neighbourhood for uniform sampling of path lengths in Gibbs entry updates
    desired_samples::Int # We now have some fields which control ouput. These will be default values for resulitng sampler
    burn_in::Int
    lag::Int
    par_info::Dict # Tag line describing each parameter
    function SimInvolutiveMcmcMergeSplit(
        ;p1=0.8, p2=0.8, β=0.0, ν_outer=2, ν_kick=2, ν_gibbs=4, desired_samples=1000, burn_in=0, lag=1
        )
        par_info = Dict()
        par_info[:p1] = "(preserved subseq size)"
        par_info[:p2] = "(diff in path length on merge/split moves)"
        par_info[:β] = "(extra prob. of Gibbs move)"
        par_info[:ν_outer] = "(neighbourhood for number of paths)"
        par_info[:ν_kick] = "(neighbourhood for kick)"
        par_info[:ν_gibbs] = "(neighbourhood for path length in Gibbs move)"  
        new(p1, p2, β, ν_outer, ν_kick, ν_gibbs, desired_samples, burn_in, lag, par_info)
    end 
end 



function Base.show(io::IO, sampler::SisInvolutiveMcmcMergeSplit)
    title = "MCMC Sampler for Spherical Interaction Sequence (SIS) Family"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(sampler))[1:6]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[7:9]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 

function Base.show(io::IO, sampler::SimInvolutiveMcmcMergeSplit)
    title = "MCMC Sampler for Spherical Interaction Multiset (SIM) Family"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(sampler))[1:6]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[7:9]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 