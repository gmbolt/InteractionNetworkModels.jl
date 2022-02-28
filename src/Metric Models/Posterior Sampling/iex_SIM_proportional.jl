function double_iex_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior,
    γ_curr::Float64,
    mcmc::SimIexInsertDeleteProportional,
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    ) 

    N = length(S_curr)  
    n_tot = sum(length, S_curr)
    m_tot = n_tot
    dist = posterior.dist
    V = posterior.V
    K_inner, K_outer = (posterior.K_inner, posterior.K_outer)
    K_in_lb, K_in_ub = (K_inner.l, K_inner.u)
    data = posterior.data
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode

    aux_mcmc = mcmc.aux_mcmc

    δ = rand(1:mcmc.ν_ed)  # Number of edits to enact 
    rem_edits = δ # Remaining edits to allocate
    j = 0 # Keeps track how many interaction have been edited 
    Z = 1
    log_ratio = 0.0

    for (i,(I_curr,I_prop)) in enumerate(zip(S_curr, S_prop))

        # If at end we just assign all remaining edits to final interaction 
        n = length(I_curr)
        if i == N 
            δ_tmp = rem_edits
        # Otherwise we sample the number of edits via rescaled Binomial 
        else 
            log_p_tmp = log(n) - log(n_tot)
            p = exp(log_p_tmp - log(Z))
            δ_tmp = rand(Binomial(rem_edits, p)) # Number of edits to perform on ith interaction 
            Z -= exp(log_p_tmp) 
        end 

        # println("   Index $i getting $δ_tmp edits...")
        # If we sampled zero edits we skip to next iteration 
        if δ_tmp == 0
            continue 
        else
            j += 1 # Increment j 
            # Make edits .... 
            @inbounds n = length(I_curr)
            d = rand(0:min(n-K_in_lb, δ_tmp))
            m = n + δ_tmp - 2*d
            m_tot += δ_tmp - 2*d

            # Catch invalid proposals
            if (m > K_in_ub)
                # Here we just reject the proposal
                for (I_prop,I_curr) in zip(S_prop,S_curr)
                    @inbounds copy!(I_prop, I_curr)
                end 
                return 0, suff_stat_curr
            end 

            ind_del = view(mcmc.ind_del, 1:d)
            ind_add = view(mcmc.ind_add, 1:(δ_tmp-d))
            vals_del = view(mcmc.vals, 1:d)


            # Sample indexing info and new entries (all in-place)
            StatsBase.seqsample_a!(1:n, ind_del)
            StatsBase.seqsample_a!(1:m, ind_add)

            # *** HERE IS DIFFERENT FROM MODEL SAMPLER ***
            # The delete_insert_informed() function does the sampling + editing 
            @inbounds log_ratio += delete_insert_informed!(
                I_prop,
                ind_del, ind_add, vals_del, 
                P)

            @inbounds mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_ratio += log(min(n-K_in_lb, δ_tmp)+1) - log(min(m-K_in_lb, δ_tmp)+1) + log(m) - log(n)

        end 

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 

    log_ratio += N*(log(n_tot) - log(m_tot))
    
    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        tmp = deepcopy(aux_data[1])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_log_lik_ratio = -γ_curr * (
        mapreduce(x -> dist(x, S_curr), + , aux_data)
        - mapreduce(x -> dist(x, S_prop), +, aux_data)
    )

    suff_stat_prop = mapreduce(x -> dist(x, S_prop), + , data)
    log_lik_ratio = -γ_curr * (
        suff_stat_prop - suff_stat_curr
    )

    log_prior_ratio = -γ_prior * (
        dist(S_prop, mode_prior) - dist(S_curr, mode_prior)
    )

    log_multinom_term = log_multinomial_ratio(S_curr, S_prop)

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio + log_multinom_term
    # println("Edit alloc proposing...")
    # println("$(S_prop)")
    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_curr[i], S_prop[i])
        end
        return 1, suff_stat_prop
    else 
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_prop[i], S_curr[i])
        end 
        return 0, suff_stat_curr
    end 
end 
