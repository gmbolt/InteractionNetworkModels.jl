using Distributions, StatsBase

export imcmc_gibbs_scan!, sample_fast

function imcmc_insert_prop_sample!(
    S_curr::BoundedInteractionSequence, 
    S_prop::BoundedInteractionSequence, 
    d::PathPseudoUniform
    )

    @assert length(S_prop) < S_prop.K_outer "Invalid insertion attempt. At max num. of interactions and trying to insert."

    i = rand(1:(length(S_curr)+1))
    m = rand(d.length_dist)

    @assert m ≤ S_prop.K_inner "Invalide insertion attempt. Interaction to be inserted is longer than K_inner"
    
    insert_interaction!(S_prop, i, 1) # Insert dummy interaction [1]
    S_prop.dims[i] = m # Indexing x[i] on next line returns a view, but we must first set the length to m as desired
    @views sample!(d.vertex_set, S_prop[i])
    S_prop.dims[i]

    log_ratio = m*log(length(d.vertex_set)) - logpdf(d.length_dist, m) 
    
    return log_ratio
end 

function imcmc_delete_prop_sample!(
    S_curr::BoundedInteractionSequence, 
    S_prop::BoundedInteractionSequence,
    d::PathPseudoUniform
)

    i = rand(1:length(S_curr))
    delete_interaction!(S_prop, i)
    m = S_curr.dims[i] # Length of path deleted 
    log_ratio = logpdf(d.length_dist, m) - m * log(length(d.vertex_set))

    return log_ratio
end 

function imcmc_gibbs_update!(
    S_curr::BoundedInteractionSequence{T}, 
    S_prop::BoundedInteractionSequence{T}, 
    i::Int, 
    model::BoundedSIS{T}, 
    mcmc::SisInvolutiveMcmcInsertDelete{T}
) where {T<:Union{Int, String}}

    n = S_curr.dims[i]
    δ = rand(1:mcmc.ν)
    d = rand(0:min(n-1,δ))
    m = n + δ - 2*d

    # Set-up views 
    ind_del = view(mcmc.ind_del, 1:d)
    ind_add = view(mcmc.ind_add, 1:(δ-d))
    vals = view(mcmc.vals, 1:(δ-d))

    # Sample indexing info and new entries (all in-place)
    StatsBase.seqsample_a!(1:n, ind_del)
    StatsBase.seqsample_a!(1:m, ind_add)
    sample!(model.V, vals)

    delete_insert_interaction!(
        S_prop, i, # enact on ith path in S_prop
        δ, d, 
        ind_del, ind_add, vals
        )
    log_ratio = log(min(n-1, δ)+1) - log(min(m-1, δ)+1) + (δ - 2*d) * log(length(model.V))


    # @show curr_dist, prop_dist
    @inbounds log_α = (
        -model.γ * (
            model.dist(model.mode, S_prop)-model.dist(model.mode, S_curr)
            ) + log_ratio
        )
    if log(rand()) < log_α
        # println("accepted")
        copy_interaction!(S_curr, S_prop, i, i)
        # @inbounds S_curr[i] = copy(S_prop[i])
        # println("$(i) value proposal $(tmp_prop) was accepted")
        return 1
    else
        # copy!(S_prop[i], S_curr[i])
        copy_interaction!(S_prop, S_curr, i, i)
        # @inbounds S_prop[i] = copy(S_curr[i])
        # println("$(i) value proposal $(tmp_prop) was rejected")
        return 0
    end 
    # @show S_curr
end 

function imcmc_gibbs_scan!(
    S_curr::BoundedInteractionSequence{T}, 
    S_prop::BoundedInteractionSequence{T}, 
    model::BoundedSIS{T}, 
    mcmc::SisInvolutiveMcmcInsertDelete{T}
    ) where {T<:Union{Int, String}}

    count = 0
    N = length(S_curr)
    for i in 1:N
        count += imcmc_gibbs_update!(S_curr, S_prop, i, model, mcmc)
    end 
    return count
end 

function sample_fast(
    mcmc::SisInvolutiveMcmcInsertDelete{T},
    model::BoundedSIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=BoundedInteractionSequence(model.mode.data)
    ) where {T<:Union{Int, String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag
    sample_out = [BoundedInteractionSequence([T[]], model.K_inner, model.K_outer)]

    # Define aliases for pointers to the storage of current vals and proposals
    S_curr = BoundedInteractionSequence(init.data)
    S_prop = BoundedInteractionSequence(init.data)

    ind = 0
    count = 0
    acc_count = 0
    gibbs_scan_count = 0
    gibbs_tot_count = 0
    gibbs_acc_count = 0 
    # is_insert = false

    # Bounds for uniform sampling
    lb(x::BoundedInteractionSequence) = max(1, length(x) - 1 )
    ub(x::BoundedInteractionSequence) = min(model.K_outer, length(x) + 1)

    # prob gibbs (effectively with ϵ=1)
    prob_gibbs = 1/(2+1) + mcmc.β

    for i in 1:req_samples

        # Gibbs scan
        if rand() < prob_gibbs
            gibbs_scan_count += 1
            gibbs_tot_count += length(S_curr)
            gibbs_acc_count += imcmc_gibbs_scan!(S_curr, S_prop, model, mcmc) # This enacts the scan, changing curr, and outputs number of accepted moves.
            copy!(sample_out[i], S_curr)
        # Else do insert or delete
        else 
            count += 1
            # If only one interaction we only insert
            if length(S_curr) == 1 
                # is_insert = true
                # ind = rand(1:(length(S_curr)+1))
                log_ratio = imcmc_insert_prop_sample!(
                    S_curr, S_prop, 
                    mcmc.path_dist
                )
                log_ratio += 0.5 # Adjust log ratio
                
            # If max mnumber of interactions we only delete
            elseif length(S_curr) == model.K_outer
                # is_insert = false
                # ind = rand(1:length(S_curr))
                log_ratio = imcmc_delete_prop_sample!(
                    S_curr, S_prop, 
                    mcmc.path_dist
                )
                log_ratio += 0.5 # Adjust log ratio

            elseif rand() < 0.5  # Coin flip for insert
                # is_insert = true
                # ind = rand(1:(length(S_curr)+1))
                log_ratio = imcmc_insert_prop_sample!(
                    S_curr, S_prop, 
                    mcmc.path_dist
                )
            else # Else delete
                # is_insert = false
                # ind = rand(1:length(S_curr))
                log_ratio = imcmc_delete_prop_sample!(
                    S_curr, S_prop, 
                    mcmc.path_dist
                )
            end 
            # println(S_curr)
            log_α = - model.γ * (
                model.dist(model.mode, S_prop) - model.dist(model.mode, S_curr)
            ) + log_ratio

            if rand() < exp(log_α)
                # if is_insert
                #     insert!(S_curr, ind, copy(S_prop[ind]))
                # else 
                #     deleteat!(S_curr, ind)
                # end 
                copy!(S_curr, S_prop)
                acc_count += 1
            else
                # if is_insert
                #     deleteat!(S_prop, ind)
                # else 
                #     insert!(S_prop, ind, copy(S_curr[ind]))
                # end 
                copy!(S_prop, S_curr)
            end 
            copy!(sample_out[i], S_curr)
        end 
    end 

    p_measures = Dict(
        "Proportion Gibbs moves" => gibbs_scan_count/(count + gibbs_scan_count),
        "Trans-dimensional move acceptance probability" => acc_count/count,
        "Gibbs move acceptance probability" => gibbs_acc_count/gibbs_tot_count
    )
    output = SisMcmcOutput(
        model, 
        sample_out[(burn_in+1):lag:end], 
        p_measures
        )

    return output
end 

