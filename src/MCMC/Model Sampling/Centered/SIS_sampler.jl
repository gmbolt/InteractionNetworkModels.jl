using Distributions

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    model::SIS{Int}, 
    mcmc::SisMcmcSplitMerge
    )

    K_in_lb, K_in_ub, K_out_lb, K_out_ub = (model.K_inner.l, model.K_inner.u, model.K_outer.l, model.K_outer.u)
    ν_td, p_ins, η, V, ind_td = (mcmc.ν_td, mcmc.p_ins, mcmc.η, model.V, mcmc.ind_td)

    prop_pointers = mcmc.prop_pointers
    curr_pointers = mcmc.curr_pointers
    γ, mode, d = (model.γ, model.mode, model.dist)

    N = length(S_curr)

    is_split = rand(Bernoulli(0.5))
    if is_split
        ub = min(ν_td, N)
        ε = rand(1:ub)
        M = N + ε
        # Catch invalid proposal (only need upper since only increasing)
        if M > K_out_ub
            return 0 
        end 
        ind_split = view(ind_td, 1:ε)
        StatsBase.seqsample_a!(1:N, ind_split)
        # @show ind_split
        log_ratio = multiple_adj_split_noisy!(
            S_prop, 
            prop_pointers, 
            ind_split, 
            p_ins, 
            η, V,
            K_in_lb, K_in_ub
        )
        log_ratio += log(ub) - log(min(floor(Int, M/2), ν_td))
    else 
        ub = min(floor(Int, N/2), ν_td)
        ε = rand(1:ub)
        M = N - ε
        # Catch invalid proposals
        if M < K_out_lb 
            return 0
        end 
        ind_merge = view(ind_td, 1:ε)
        StatsBase.seqsample_a!(1:M, ind_merge)
        # @show ind_merge
        log_ratio = multiple_adj_merge_noisy!(
            S_prop,
            prop_pointers, 
            ind_merge, 
            p_ins, η, V,
            K_in_lb, K_in_ub
        )
        log_ratio += log(ub) - log(min(M, ν_td))
    end 

    log_α = - γ * (d(mode, S_prop) - d(mode, S_curr)) + log_ratio

    if log(rand()) < log_α
        # We accept (make S_curr into S_prop)
        if is_split 
            # Copy split to S_curr
            live_index = 0
            for i in ind_split
                copy!(S_curr[i+live_index], S_prop[i+live_index])
                I_new = popfirst!(curr_pointers)
                copy!(I_new, S_prop[i+live_index+1])
                insert!(S_curr, i+live_index+1, I_new)
            end 
        else 
            # Copy merge to S_curr
            for i in ind_merge
                I_old = popat!(S_curr, i+1)
                pushfirst!(curr_pointers, I_old)
                copy!(S_curr[i], S_prop[i])
            end 
        end 
    else 
        # We reject
        if is_split 
            # Undo split 
            
        else 
            # Undo merge 
        end 
    end 
end 