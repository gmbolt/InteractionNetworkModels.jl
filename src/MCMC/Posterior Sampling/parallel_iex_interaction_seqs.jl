using Distributed, ParallelDataTransfer

export piex_mcmc_mode, test1, test2


# function log_lik(
#     data::InteractionSequenceSample,
#     S::InteractionSequence, 
#     γ::Real,
#     d::InteractionSeqDistance
# )
#     return mapreduce(x->-γ*d(x, S), +, data)

# end 

# function plog_lik(
#     data::InteractionSequenceSample, 
#     S::InteractionSequence, 
#     γ::Real,
#     d::InteractionSeqDistance,
#     split_inds::Vector{UnitRange})
#     out = Distributed.pmap(split_inds) do index
#         return log_lik(data[index], S, γ, d)
#     end 
#     out
# end 


# Auxiliary ratio helper functions 
# --------------------------------
# 1. Evaluate log ratio term for single auxiliary data point, here γ_fixed, d_fixed, S_curr, S_prop
#    are all global variables within the package 

# @everywhere include("../../aliases.jl")

function aux_log_lik_ratio_mode(
    x::InteractionSequence,
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence
    )
    
    return -γ_fix * (d_fix(x, S_curr) - d_fix(x, S_prop))

end 
# 2. Simulate n auxiliary data points and evluate the sum of log ratios
function aux_log_lik_ratio_mode(
    n::Int,
    S_curr::InteractionSequence,
    S_prop::InteractionSequence
    ) 

    data = draw_sample(mcmc_sampler, model, desired_samples=n)
    mapreduce(x->log_ratio_aux_mode(x, S_curr, S_prop), +, data)

end 

function aux_log_lik_ratio_mode(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence,
    split::Vector{Int}
    )
    return pmap(x->aux_log_lik_ratio_mode(x, S_curr, S_prop), split)
end 

# Log likelihood ratio helper functions 
# -------------------------------------

# 1. Single data point
function log_lik_ratio(
    x::InteractionSequence,
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence)  

    return -γ_fix * (d_fix(x, S_prop) - d_fix(x, S_curr))

end 

# 2. Subsequence of datapoints, data_fixed is a global variable in the package storing the 
#    observed data. 
function log_lik_ratio(
    ind::UnitRange,
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence)

    return mapreduce(x->log_lik_ratio(x, S_curr, S_prop), +, data_fix[ind])

end 

function log_lik_ratio(
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence,
    split_inds::Vector{UnitRange{Int}}
    )
    return pmap(x->log_lik_ratio(x, S_curr, S_prop), split_inds)

end 
function piex_mcmc_within_gibbs_update!(
    posterior::SisPosterior{T},
    S_curr::InteractionSequence,
    i::Int, 
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    split::Vector{Int},
    data_split_inds::Vector{UnitRange{Int}}
) where {T<:Union{Int, String}}

    S_prop = deepcopy(S_curr)

    @inbounds n = length(S_curr[i])

    δ = rand(1:ν)
    d = rand(0:min(n-1,δ))

    if T==Int
        I_prop, log_ratio = imcmc_prop_sample_edit_informed(S_curr[i]::Path, δ, d, P)
    else 
        I_prop, log_ratio = imcmc_prop_sample_edit_informed(S_curr[i]::Path, δ, d, P, vertex_map, vertex_map_inv)
    end 

    @inbounds S_prop[i] = I_prop

    # @show S_curr, S_prop

    # Sample auxiliary data 
    # aux_model = SIS(S_prop, γ_curr, posterior.dist, posterior.V, posterior.K_inner, posterior.K_outer)
    # draw_sample!(aux_data, mcmc_sampler, aux_model)
    
    # log_lik_ratio = - γ_curr * (
    #         sum_of_dists(posterior.data, S_prop, posterior.dist)
    #         -sum_of_dists(posterior.data, S_curr, posterior.dist)
    #     )
    log_lik_ratio_term = sum(pmap(log_lik_ratio, data_split_inds))
    # @show aux_data
    # Run burn-in to get initial values
    # aux_log_lik_ratio= plog_aux_term_mode_update(
    #     mcmc_sampler, aux_model, 
    #     split,
    #     S_curr, S_prop
    #     )
    log_aux_ratio_term = sum(pmap(log_ratio_aux_mode, split))
    # @show log_lik_ratio, aux_log_lik_ratio
    log_α = (
        posterior.S_prior.γ *(
            sum_of_dists(posterior.S_prior, S_curr)
            - sum_of_dists(posterior.S_prior, S_prop)
        )
        + log_lik_ratio_term + log_aux_ratio_term + log_ratio
    ) 
    # println("Gibbs:")
    if log(rand()) < log_α
        @inbounds S_curr[i] = copy(S_prop[i])
        # println("$(i) value proposal $(tmp_prop) was accepted")
        return 1
        
    else
        # println("$(i) value proposal $(tmp_prop) was rejected")
        return 0
    end 
end 


function piex_mcmc_within_gibbs_scan!(
    posterior::SisPosterior{T},
    S_curr::InteractionSequence,
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    mcmc_sampler::SisMcmcSampler, # Sampler for auxiliary variables 
    split::Vector{Int}, # Split of samples over processors
    data_split_inds::Vector{UnitRange}
    ) where {T<:Union{Int, String}}

    N = length(S_curr)
    count = 0
    for i in 1:N
        count += piex_mcmc_within_gibbs_update!(
            posterior, 
            S_curr,
            i,
            ν,
            P, vertex_map, vertex_map_inv, 
            mcmc_sampler,
            split,
            data_split_inds)
    end 
    return count
end 

function piex_mcmc_mode(
    posterior::SisPosterior{T},
    mcmc_sampler::SisMcmcSampler,
    γ_fixed::Float64;
    S_init::Vector{Path{T}}=sample_frechet_mean(posterior.data, poseterior.dist),
    desired_samples::Int=100, # MCMC parameters...
    burn_in::Int=100,
    lag::Int=1,
    ν::Int=1, 
    path_dist::PathDistribution{T}=PathPseudoUniform(posterior.V, TrGeometric(0.6, 1, 10)),
    β = 0.0,
    α = 0.0
    ) where {T<:Union{Int, String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag

    iter = Progress(req_samples, 1, "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    
    # Intialise 
    S_sample = Vector{Vector{Path{T}}}(undef, req_samples) # Storage of samples
    S_curr = S_init 

    # Send values to workers 
    sendto(
        workers(), 
        γ_fix=γ_fixed, # Fixed value for dispersion
        d_fix=posterior.dist, # Fixed value for distance (will be stored locally on each worker, so storage associated with distance will be local to worker too)
        data_fix=posterior.data, # Data set
        mcmc_sampler=mcmc_sampler # Mcmc sampler (similar to distance, will be local so associated storage will also be local)
        )

    count = 0 
    acc_count = 0
    gibbs_scan_count = 0
    gibbs_tot_count = 0
    gibbs_acc_count = 0 

    # Splitting over processors
    mcmc_split = get_split(posterior.sample_size, Distributed.nworkers())
    data_split_inds = get_split_ind_iters(posterior.sample_size, Distributed.nworkers())

    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior.data, α)

    # Bounds for uniform sampling of number of interactions
    lb(x::Vector{Path{T}}) = max( ceil(Int, length(x)/2), length(x) - ν_outer )
    ub(x::Vector{Path{T}}) = min(posterior.K_outer, 2*length(x), length(x) + ν_outer)

    probs_gibbs = 1/(2+1) + β
    
    # Run one MCMC burn_in at S_curr to get an aux_end value (the final value from MCMC chain)

    for i in 1:req_samples
        
        if rand() < probs_gibbs
            gibbs_scan_count += 1
            gibbs_tot_count += length(S_curr)
            gibbs_acc_count += piex_mcmc_within_gibbs_scan!(
                posterior, # Target
                S_curr,
                ν, 
                P, vmap, vmap_inv, 
                mcmc_sampler,
                mcmc_split, data_split_inds) # Gibbs sampler parameters
            S_sample[i] = deepcopy(S_curr)
            next!(iter)
        
        else 
            count +=  1

            if length(S_curr) == 1
                S_prop, log_ratio = imcmc_insert_prop_sample(
                    S_curr, path_dist
                )
                log_ratio *= 0.5
            elseif length(S_curr) == posterior.K_outer
                S_prop, log_ratio = imcmc_delete_prop_sample(
                    S_curr, path_dist
                )
                log_ratio *= 0.5
            elseif rand() < 0.5 # w.p. 1/2 insert a new interaction randomly
                S_prop, log_ratio = imcmc_insert_prop_sample(
                    S_curr,
                    path_dist
                )
            else # w.p. 1/2 delete an interaction randomly
                S_prop, log_ratio = imcmc_delete_prop_sample(
                    S_curr, 
                    path_dist
                )
            end 

            # Define auxiliary model on all workers
            # sendto(workers(), aux_model = SIS(S_prop, γ_curr, posterior.dist, posterior.V, posterior.K_inner, posterior.K_outer))
            # draw_sample!(aux_data, mcmc_sampler, aux_model)
            # aux_data = pmap()

            # Accept reject
            # log_lik_ratio = - γ_curr * (
            # sum_of_dists(posterior.data, S_prop, posterior.dist)
            # -sum_of_dists(posterior.data, S_curr, posterior.dist)
            #     )
            log_lik_ratio_term = log_lik_ratio(S_curr, S_prop, data_split_inds)
            # aux_log_lik_ratio = -γ_curr * (
            #         sum_of_dists(aux_data, S_curr, posterior.dist)
            #         - sum_of_dists(aux_data, S_prop, posterior.dist)
            #     )

            # aux_log_lik_ratio = plog_aux_term_mode_update(
            #     mcmc_sampler, aux_model, 
            #     psplit,
            #     S_curr, S_prop
            #     )
            aux_log_lik_ratio_term = aux_log_lik_ratio_mode(S_curr, S_prop, mcmc_split)

            log_α = (
                posterior.S_prior.γ *(
                    d_fix(posterior.S_prior, S_curr)
                    - d_fix(posterior.S_prior, S_prop)
                )
                + log_lik_ratio_term + aux_log_lik_ratio_term + log_ratio
            ) 
            # println("Transdim:")
            # @show log_lik_ratio, aux_log_lik_ratio

            if log(rand()) < log_α
                S_sample[i] = deepcopy(S_prop)
                S_curr = deepcopy(S_prop)
                acc_count += 1
            else 
                S_sample[i] = deepcopy(S_curr)
            end 
            next!(iter)
        end 

    end 
    p_measures = Dict(
        "Proportion Gibbs moves" => gibbs_scan_count/(count + gibbs_scan_count),
        "Trans-dimensional move acceptance probability" => acc_count/count,
        "Gibbs move acceptance probability" => gibbs_acc_count/gibbs_tot_count
    )
    output = SisPosteriorModeConditionalMcmcOutput(
        γ_fixed, 
        S_sample[(burn_in+1):lag:end],
        posterior.dist,
        posterior.S_prior, 
        posterior.data,
        p_measures
    )
    return output
end 



