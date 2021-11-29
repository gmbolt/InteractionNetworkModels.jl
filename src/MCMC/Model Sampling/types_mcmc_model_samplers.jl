export SisMcmcSampler, SimMcmcSampler, SpfMcmcSampler
export SisMcmcInsertDeleteEdit, SisMcmcInsertDeleteGibbs
export SimMcmcInsertDeleteEdit, SimMcmcInsertDeleteGibbs
export SisMcmcInitialiser, SisInitMode, SisInitRandEdit, get_init
export SisMcmcInitialiser, SimInitMode
export SpfInvolutiveMcmcCentSubseq, SpfInvolutiveMcmcEdit


# ====================
#   SPF 
# ====================

# Mcmc samplers for a metric-model for paths (sequences)

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
#      SIS 
# =====================================

# Initilisers 
# -----------

# These can be passed to mcmc samplers to determine the defualt initialisation scheme. 

""" 
Abstract type representing initialisation schemes for SIS model samplers. 
"""
abstract type SisMcmcInitialiser end


"""
`SisInitMode <: SisMcmcInitialiser` - this is a MCMC initialisation scheme for SIS model samplers which starts the MCMC chain at the model mode by default.
"""
struct SisInitMode <: SisMcmcInitialiser
    function SisInitMode()
        return new() 
    end 
end 

function get_init(
    model::SIS, 
    initiliaser::SisInitMode
    )
    return model.mode
end 

struct SisInitRandEdit <: SisMcmcInitialiser
    δ::Int
    function SisInitRandEdit(δ::Int) 
        return new(δ)
    end 
end 

function get_init(
    initialiser::SisInitRandEdit,
    S::InteractionSequence,
    V::AbstractArray,
    K_inner::Int
    )

    δ = initialiser.δ 
    S_init = S
    N = length(S)

    ind_del = zeros(Int, δ)
    ind_add = zeros(Int, δ)
    vals = zeros(Int, δ)

    rem_edits = δ

    for i in 1:N 
        if i == N 
            δ_tmp = rem_edits
        else 
            p = 1/(N=i+1)
            δ_tmp = rand(Binomial(rem_edits, p))
        end 

        if δ_tmp == 0
            continue 
        else

            
            n = length(model.mode[i])
            d = rand(max(0, ceil(Int, (n + δ_tmp + - K_inner)/2)):min(n, δ_tmp))
            m = n + δ_tmp - 2*d

            ind_del_v = view(ind_del, 1:d)
            ind_add_v = view(ind_add, 1:(δ_tmp-d))
            vals_v = view(vals, 1:(δ_tmp-d))

            StatsBase.seqsample_a!(1:n, ind_del_v)
            StatsBase.seqsample_a!(1:m, ind_add_v)
            sample!(V, vals)

            delete_insert!(S_init[i], ind_del_v, ind_add_v, vals_v)

        end 

        rem_edits -= δ_tmp 

        if rem_edits == 0 
            break 
        end 
    end 

    return S_init

end 
function get_init(
    initialiser::SisInitRandEdit,
    model::SIS
    )
    return get_init(initialiser, model.mode, model.V, model.K_inner)
end 


abstract type SisMcmcSampler end 

# From Models 

# Insert-Delete 
# -------------

struct SisMcmcInsertDeleteGibbs{T<:Union{Int,String}} <: SisMcmcSampler
    ν_gibbs::Int   # Maximum number of edit ops
    ν_trans_dim::Int  # Maximum increase or decrease in number of interactions
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    β::Real  # Extra probability of Gibbs move
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::SisMcmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions in Gibbs scan
    ind_add::Vector{Int} # Storage for indexing of additions in Gibbs scan
    vals::Vector{T} # Storage for valuse to insert in Gibbs scan
    ind_trans_dim::Vector{Int} # Storage of where to insert/delete 
    function SisMcmcInsertDeleteGibbs(
        path_dist::PathDistribution{S};
        K=100,
        ν_gibbs=4, ν_trans_dim=2,  β=0.6,
        desired_samples=1000, lag=1, burn_in=0,
        init=SisInitMode()
        ) where {S<:Union{Int, String}}
        curr_pointers = [S[] for i in 1:K]
        prop_pointers = [S[] for i in 1:K]
        ind_del = zeros(Int, ν_gibbs)
        ind_add = zeros(Int, ν_gibbs)
        vals = zeros(Int, ν_gibbs)
        ind_trans_dim = zeros(Int, ν_trans_dim)
        par_info = Dict()
        par_info[:ν_gibbs] = "(maximum number of edit operations in iMCMC-within-Gibbs conditional updates)"
        par_info[:ν_trans_dim] = "(maximum number of interaction insertions or deletions)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of Gibbs scan)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν_gibbs, ν_trans_dim, path_dist, β, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals, ind_trans_dim
            )
    end 
end 

SisMcmcInsertDeleteGibbs(
    model::SIS;
    K=100,
    ν=4, β=0.0,
    desired_samples=1000, lag=1, burn_in=0, init=SisInitMode()
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


struct SisMcmcInsertDeleteEdit{T<:Union{Int,String}} <: SisMcmcSampler
    ν_edit::Int  # Maximum number of edit operations
    ν_trans_dim::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::SisMcmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{T} # Storage for values to insert in interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_trans_dim::Vector{Int} # Storage of where to insert/delete 
    function SisMcmcInsertDeleteEdit(
        path_dist::PathDistribution{S};
        K=100,
        ν_edit=2, ν_trans_dim=2, β=0.4,
        desired_samples=1000, lag=1, burn_in=0, init=SisInitMode()
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
        par_info[:ν_trans_dim] = "(maximum increase/decrease in dimension)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν_edit, ν_trans_dim, β, 
            path_dist, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_trans_dim
            )
    end 
end 

function Base.show(io::IO, sampler::SisMcmcInsertDeleteEdit)
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

# ============================
#  SIM 
# ============================

# Intialisers 
# -----------

# These are fed into samplers to define an intialisation scheme. Though in general we just intialise at the mode via SimInitMode(). 

""" 
Abstract type representing initialisation schemes for SIM model samplers. 
"""
abstract type SimMcmcInitialiser end

"""
`SisInitMode <: SisMcmcInitialiser` - this is a MCMC initialisation scheme for SIS model samplers which starts the MCMC chain at the model mode by default.
"""
struct SimInitMode <: SimMcmcInitialiser
    function SimInitMode()
        return new() 
    end 
end 

function get_init(
    model::SIM, 
    initiliaser::SimInitMode
    )
    return model.mode
end 

# Samplers 
# --------

# Abstract type (all samplers are subtype of this)

abstract type SimMcmcSampler end 

# Edit Allocation Sampler 
# -----------------------

struct SimMcmcInsertDeleteEdit{T<:Union{Int,String}} <: SimMcmcSampler
    ν_edit::Int  # Maximum number of edit operations
    ν_trans_dim::Int  # Maximum change in outer dimension
    β::Real  # Probability of trans-dimensional move
    path_dist::PathDistribution{T}  # Distribution used to introduce new interactions
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    init::SimMcmcInitialiser
    par_info::Dict
    curr_pointers::InteractionSequence{T} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{T} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{T} # Storage for values to insert in interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_trans_dim::Vector{Int} # Storage of where to insert/delete 
    function SimMcmcInsertDeleteEdit(
        path_dist::PathDistribution{S};
        K=100,
        ν_edit=2, ν_trans_dim=2, β=0.4,
        desired_samples=1000, lag=1, burn_in=0, init=SimInitMode()
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
        par_info[:ν_trans_dim] = "(maximum increase/decrease in dimension)"
        par_info[:path_dist] = "(path distribution for insertions)"
        par_info[:β] = "(probability of update move)"
        par_info[:K] = "(maximum number of interactions, used to initialise storage)"

        new{S}(
            ν_edit, ν_trans_dim, β, 
            path_dist, K,
            desired_samples, burn_in, lag, init,
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_trans_dim
            )
    end  
end 

function Base.show(io::IO, sampler::SimMcmcInsertDeleteEdit)
    title = "MCMC Sampler for SIM Models via Edit Allocation + Insert/Delete Moves"
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

# Gibbs Scan Sampler
# ------------------

struct SimMcmcInsertDeleteGibbs{T<:Union{Int,String}} <: SimMcmcSampler
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
    function SimMcmcInsertDeleteGibbs(
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
