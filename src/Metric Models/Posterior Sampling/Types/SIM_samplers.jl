export SimPosteriorSampler, SimIexInsertDelete, SimIexInsertDeleteProportional
export SimIexInsertDeleteDependent, SimIexInsertDeleteWithKick
export SimIexSplitMerge

abstract type SimPosteriorSampler end 

struct SimIexInsertDelete <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε::Float64
    aux_mcmc::SimMcmcSampler
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
    function SimIexInsertDelete(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε=0.1,
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
            ind_update, 
            ind_td_add, ind_td_del
        )
    end 
end 

struct SimIexSplitMerge <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε::Float64
    aux_mcmc::SimMcmcSampler
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    store::InteractionSequence{Int} # Temporary storage between for use in move
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{Int} # Storage for values where are deleted from interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td_split::Vector{Int} # Storage for split/merge locations
    ind_td_merge::Vector{Int} # Storage for split/merge locations
    function SimIexSplitMerge(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε=0.1,
        desired_samples=1000, lag=1, burn_in=0
        ) 
        curr_pointers = [Int[] for i in 1:K]
        prop_pointers = [Int[] for i in 1:K]
        store = Vector{Int}[]
        ind_del = zeros(Int, ν_ed)
        ind_add = zeros(Int, ν_ed)
        vals = zeros(Int, ν_ed)
        ind_update = zeros(Int, ν_ed)
        ind_td_split = zeros(Int, ν_td)
        ind_td_merge = zeros(Int, 2*ν_td) # Need to index double 
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
            curr_pointers, prop_pointers, store,
            ind_del, ind_add, vals,
            ind_update, 
            ind_td_split, ind_td_merge
        )
    end 
end 


struct SimIexInsertDeleteGibbs <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε::Float64
    aux_mcmc::SimMcmcSampler
    K::Int # Max number of interactions (used to determined how many pointers to store interactions)
    desired_samples::Int  # Final three set default values for MCMC samplers 
    burn_in::Int
    lag::Int
    par_info::Dict
    count::Vector{Int}      # Counts for attempted moves
    acc_count::Vector{Int}  # Counts for accepted moves
    curr_pointers::InteractionSequence{Int} # Storage for prev value in MCMC
    prop_pointers::InteractionSequence{Int} # Storage for curr value in MCMC
    ind_del::Vector{Int} # Storage for indexing of deletions of interactions
    ind_add::Vector{Int} # Storage for indexing of additions of interactions
    vals::Vector{Int} # Storage for values where are deleted from interactions
    ind_update::Vector{Int} # Storage of which values have been updated
    ind_td_add::Vector{Int} # Storage of where to insert/delete 
    ind_td_del::Vector{Int} # Storage of where to insert/delete 
    function SimIexInsertDeleteGibbs(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε=0.1,
        desired_samples=1000, lag=1, burn_in=0
        ) 
        count, acc_count = (zeros(Int, 3), zeros(Int, 3))
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
            count, acc_count,
            curr_pointers, prop_pointers, 
            ind_del, ind_add, vals,
            ind_update, 
            ind_td_add, ind_td_del
        )
    end 
end 

get_summary_measures(mcmc::SimMcmcInsertDeleteGibbs) = Dict(
    "Insert/Delete Acc. Prob." => mcmc.acc_count[1]/mcmc.count[1],
    "Flip Acc. Prob." => mcmc.acc_count[2]/mcmc.count[2],
    "Trans-dim Acc. Prob." => mcmc.acc_count[3]/mcmc.count[3]
)


struct SimIexInsertDeleteProportional <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε::Float64
    aux_mcmc::SimMcmcSampler
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
    function SimIexInsertDeleteProportional(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε=0.1,
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
            ind_update, 
            ind_td_add, ind_td_del
            )
    end 
end 

struct SimIexInsertDeleteDependent <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε_ed::Float64
    ε_td::Float64
    aux_mcmc::SimMcmcSampler
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
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SimIexInsertDeleteDependent(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε_ed=0.1, ε_td=2.0,
        desired_samples=1000, lag=1, burn_in=0
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
            ε_ed, ε_td,
            aux_mcmc, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td
            )
    end 
end 


struct SimIexInsertDeleteWithKick <: SimPosteriorSampler
    ν_ed::Int  # Maximum number of edit operations
    ν_td::Int  # Maximum increase/decrease in dimension
    β::Real  # Probability of trans-dimensional move
    α::Float64  # How much vertex proposal is informed by data (0.0 max and ∞ is uniform over vertice)
    len_dist::DiscreteUnivariateDistribution  # Distribution used to introduce new interactions
    ε_ed::Float64
    ε_td::Float64
    ε_kick::Float64
    aux_mcmc::SimMcmcSampler
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
    ind_td::Vector{Int} # Storage of where to insert/delete 
    function SimIexInsertDeleteWithKick(
        aux_mcmc::SimMcmcSampler;
        K=100,
        ν_ed=2, ν_td=1 , β=0.7, α=0.0, len_dist=TrGeometric(0.9,1,K),
        ε_ed=0.1, ε_td=1.0, ε_kick=ε_td/4, 
        desired_samples=1000, lag=1, burn_in=0
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
            ε_ed, ε_td, ε_kick,
            aux_mcmc, K,
            desired_samples, burn_in, lag, 
            par_info,
            curr_pointers, prop_pointers, ind_del, ind_add, vals,
            ind_update, ind_td
            )
    end 
end 
