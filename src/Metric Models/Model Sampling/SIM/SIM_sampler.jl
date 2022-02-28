using Distributions, StatsBase

# Edit Allocation Move 
# --------------------

function imcmc_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence, 
    model::SIM, 
    mcmc::T
    ) where {T<:SimMcmcSampler}

    N = length(S_curr)  
    K_in_lb = model.K_inner.l
    K_in_ub = model.K_inner.u
    δ = rand(1:mcmc.ν_ed)  # Number of edits to enact 
    rem_edits = δ # Remaining edits to allocate
    len_diffs = 0
    j = 0 # Keeps track how many interaction have been edited 
    log_prod_term = 0.0 

    # println("Making $δ edits in total...")

    for i = 1:N

        # If at end we just assign all remaining edits to final interaction 
        if i == N 
            δ_tmp = rem_edits
        # Otherwise we sample the number of edits via rescaled Binomial 
        else 
            p = 1/(N-i+1)
            δ_tmp = rand(Binomial(rem_edits, p)) # Number of edits to perform on ith interaction 
        end 

        # println("   Index $i getting $δ_tmp edits...")
        # If we sampled zero edits we skip to next iteration 
        if δ_tmp == 0
            continue 
        else
            j += 1 # Increment j 
            # Make edits .... 
            @inbounds n = length(S_curr[i])
            # a,b = (lb(n, δ_tmp, model), ub(n, δ_tmp))
            d = rand(0:min(n,δ_tmp))
            m = n + δ_tmp - 2*d

            # Catch invalid proposals
            if (m < K_in_lb) | (m > K_in_ub)
                # Here we just reject the proposal
                for i in 1:N
                    @inbounds copy!(S_prop[i], S_curr[i])
                end 
                return 0 
            end 

            # tot_dels += d
            # println("       Deleting $d and adding $(δ_tmp-d)")
            ind_del = view(mcmc.ind_del, 1:d)
            ind_add = view(mcmc.ind_add, 1:(δ_tmp-d))
            vals = view(mcmc.vals, 1:(δ_tmp-d))

            # println("           ind_del: $ind_del ; ind_add: $ind_add")

            # Sample indexing info and new entries (all in-place)
            StatsBase.seqsample_a!(1:n, ind_del)
            StatsBase.seqsample_a!(1:m, ind_add)
            sample!(model.V, vals)

            @inbounds delete_insert!(S_prop[i], ind_del, ind_add, vals)

            @inbounds mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_prod_term += log(min(n, δ_tmp)+1) - log(min(m, δ_tmp)+1)
            len_diffs += m-n  # How much bigger the new interaction is 
        end 

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 

    # # Add final part of log_ratio term
    log_ratio = log(length(model.V)) * len_diffs + log_prod_term
    # log_ratio = log_dim_diff + log_prod_term
    log_lik_ratio = -model.γ * (
        model.dist(model.mode, S_prop)-model.dist(model.mode, S_curr)
        )

    log_multinom_ratio_term = log_multinomial_ratio(S_curr, S_prop)
    # @show log_dim_diff, log_prod_term, log_lik_ratio
        
    # Log acceptance probability
    log_α = log_lik_ratio + log_ratio + log_multinom_ratio_term

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_curr[i], S_prop[i])
        end
        return 1 
    else 
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_prop[i], S_curr[i])
        end 
        return 0 
    end 
end 

# Trans-dimensional Move 
# ----------------------

# function imcmc_multi_insert_prop_sample!(
#     S_curr::InteractionSequence, 
#     S_prop::InteractionSequence,
#     mcmc::SimMcmcInsertDelete,
#     ind::AbstractVector{Int},
#     V::UnitRange,
#     K_in_ub::Int
#     ) 

#     prop_pointers = mcmc.prop_pointers
#     ν_td = mcmc.ν_td
#     N = length(S_curr)
#     len_dist = mcmc.len_dist

#     log_ratio = 0.0 
#     for i in ind 
#         migrate!(S_prop, prop_pointers, i, 1)
#         rand!(S_prop[i], path_dist)
#         log_ratio += - logpdf(path_dist, S_prop[i])
#     end 
#     log_ratio += log(ν_td) - log(min(ν_td,N)) 
#     return log_ratio 

# end 

# function imcmc_multi_delete_prop_sample!(
#     S_curr::InteractionSequence, 
#     S_prop::InteractionSequence, 
#     mcmc::SimMcmcInsertDelete,
#     ind::AbstractVector{Int}
#     ) 

#     prop_pointers = mcmc.prop_pointers
#     ν_td = mcmc.ν_td
#     N = length(S_curr)
#     path_dist = mcmc.path_dist

#     log_ratio = 0.0

#     for i in Iterators.reverse(ind)
#         migrate!(prop_pointers, S_prop, 1, i)
#         log_ratio += logpdf(path_dist, S_curr[i])
#     end 

#     log_ratio += log(min(ν_td,N)) - log(ν_td)
#     return log_ratio

# end 

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence, 
    model::SIM, 
    mcmc::T
    ) where {T<:SimMcmcSampler}

    K_out_lb = model.K_outer.l
    K_out_ub = model.K_outer.u
    K_in_ub = model.K_inner.u
    V = model.V
    ν_td = mcmc.ν_td
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    is_insert = rand(Bernoulli(0.5))
    if is_insert
        ε = rand(1:ν_td) # How many to insert 
        # Catch invalid proposal (ones which have zero probability)
        if (N + ε) > K_out_ub
            # Make no changes and imediately reject  
            return 0  
        end 
        ind_tr_dim = view(mcmc.ind_td, 1:ε) # Storage for where to insert 
        StatsBase.seqsample_a!(1:(N+ε), ind_tr_dim) # Sample where to insert 
        log_ratio += imcmc_multi_insert_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            ind_tr_dim,
            V, K_in_ub
            ) # Enact move and catch log ratio term 
    else 
        ε = rand(1:min(ν_td, N)) # How many to delete
        # Catch invalid proposal (would go to empty inter seq)
        if (N - ε) < K_out_lb
            return 0 
        end  
        ind_tr_dim = view(mcmc.ind_td, 1:ε) # Storage
        StatsBase.seqsample_a!(1:N, ind_tr_dim) # Sample which to delete 
        log_ratio += imcmc_multi_delete_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            ind_tr_dim,
            V
            ) # Enact move and catch log ratio 
    end 

    log_multinom_term = log_multinomial_ratio(S_curr, S_prop)
    # Now do accept-reject step 
    log_α = - model.γ * (
        model.dist(model.mode, S_prop) - model.dist(model.mode, S_curr)
    ) + log_ratio + log_multinom_term

    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
    if log(rand()) < log_α
        if is_insert
            for i in ind_tr_dim
                migrate!(S_curr, curr_pointers, i, 1)
                @inbounds copy!(S_curr[i], S_prop[i])
            end 
        else 
            for i in Iterators.reverse(ind_tr_dim)
            migrate!(curr_pointers , S_curr, 1, i)
            end 
        end 
        return 1
    else 
        # Here we must delete the interactions which were added to S_prop
        if is_insert
            for i in Iterators.reverse(ind_tr_dim)
                migrate!(prop_pointers, S_prop, 1, i)
            end 
        # Or reinsert the interactions which were deleted 
        else 
            for i in ind_tr_dim
                migrate!(S_prop, prop_pointers, i, 1)
                @inbounds copy!(S_prop[i], S_curr[i])
            end 
        end 
        return 0
    end 

end 

# Sampler Functions 
# -----------------
"""
    draw_sample!(
        sample_out::InteractionSequenceSample, 
        mcmc::SimMcmcInsertDelete, 
        model::SIM;
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::InteractionSequence=get_init(model, mcmc.init)
    )

Draw sample in-place from given SIM model `model::SIM` via MCMC algorithm with edit allocation and interaction insertion/deletion, storing output in `sample_out::InteractionSequenceSample`. 

Accepts keyword arguments to change MCMC output, including burn-in, lag and initial values. If not given, these are set to the default values of the passed MCMC sampler `mcmc::SimMcmcInsertDelete`.
"""
function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::T,
    model::SIM;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) where {T<:SimMcmcSampler}



    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    β = mcmc.β

    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for i in 1:length(init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], init[i])
        copy!(S_prop[i], init[i])
    end 

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 0 # Keeps track all samples (included lags and burn_ins) 
    upd_count = 0
    upd_acc_count = 0
    tr_dim_count = 0 
    tr_dim_acc_count = 0

    while sample_count ≤ length(sample_out)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end 
        # W.P. do update move (accept-reject done internally by function call)
        if rand() < β
            upd_acc_count += imcmc_multinomial_edit_accept_reject!(
                S_curr, S_prop, 
                model, mcmc
                )
            upd_count += 1
        # Else do trans-dim move. We will do accept-reject move here 
        else 
            tr_dim_acc_count += imcmc_trans_dim_accept_reject!(
                S_curr, S_prop, 
                model, mcmc
            )
            tr_dim_count += 1
        end 
    end 
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return (
                upd_count, upd_acc_count,
                tr_dim_count, tr_dim_acc_count
            )
end 

function draw_sample(
    mcmc::T, 
    model::SIM;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) where {T<:SimMcmcSampler}

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 

function (mcmc::T where {T<:SimMcmcSampler})(
    model::SIM;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    )

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    # @show sample_out
    (
        update_count, update_acc_count, 
        trans_dim_count, trans_dim_acc_count
        ) = draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)

    p_measures = Dict(
            "Proportion Update Moves" => update_count/(update_count+trans_dim_count),
            "Update Move Acceptance Probability" => update_acc_count / update_count,
            "Trans-Dimensional Move Acceptance Probability" => trans_dim_acc_count / trans_dim_count
        )
    output = SimMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 