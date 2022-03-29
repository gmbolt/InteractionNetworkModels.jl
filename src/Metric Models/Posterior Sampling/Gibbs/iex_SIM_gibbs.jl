
function imcmc_gibbs_insert_delete_update!(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence,
    γ_curr::Float64,
    i::Int,
    posterior::SIM, 
    mcmc::SimMcmcInsertDeleteGibbs,
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    )
    V = posterior.V 

    @inbounds I_tmp = S_prop[i]
    n = length(I_tmp)

    δ = rand(1:mcmc.ν_ed)

    d = rand(0:min(n,δ))
    m = n + δ_tmp - 2*d

    ind_del = view(mcmc.ind_del, 1:d)
    ind_add = view(mcmc.ind_add, 1:(δ_tmp-d))
    vals_del = view(mcmc.vals, 1:d)

    # Sample indexing info and new entries (all in-place)
    StatsBase.seqsample_a!(1:n, ind_del)
    StatsBase.seqsample_a!(1:m, ind_add)

    # *** HERE IS DIFFERENT FROM MODEL SAMPLER ***
    # The delete_insert_informed() function does the sampling + editing 
    log_ratio = delete_insert_informed!(
        I_tmp,
        ind_del, ind_add, vals_del, 
        P
    )

    log_ratio += log(min(n, δ)+1) - log(min(m, δ)+1)

    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        @inbounds tmp = deepcopy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_log_lik_ratio = -γ_curr * (
        sum(x -> dist(x, S_curr) for x in  aux_data)
        - sum(x -> dist(x, S_prop) for x in aux_data)
    )

    suff_stat_prop = sum(x -> dist(x, S_prop) for x in data)
    log_lik_ratio = -γ_curr * (
        suff_stat_prop - suff_stat_curr
    )

    log_multinom_ratio_term = log_multinomial_ratio(S_curr, S_prop)

    log_prior_ratio = -γ_prior * (
        dist(S_prop, mode_prior) - dist(S_curr, mode_prior)
    )

    # Log acceptance probability
    log_α = (
        log_lik_ratio + 
        log_prior_ratio + 
        aux_log_lik_ratio + 
        log_ratio +
        log_multinom_ratio_term
    )

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        copy!(S_curr[i], I_tmp)
        return 1, suff_stat_prop
    else 
        copy!(I_tmp, S_curr[i])
        return 0, suff_stat_curr
    end 

end 


function imcmc_gibbs_flip_update!(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence,
    γ_curr::Float64,
    i::Int,
    posterior::SIM, 
    mcmc::SimMcmcInsertDeleteGibbs,
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    )
    V = posterior.V 

    @inbounds I_tmp = S_prop[i]
    n = length(I_tmp)

    δ = rand(1:min(mcmc.ν_ed, n))

    ind = view(mcmc.ind_add, 1:δ)

    # Sample entries to flip
    StatsBase.seqsample_a!(1:n, ind)

    log_ratio = flip_informed_excl!(
        I_tmp,
        ind, 
        P
    )

    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
        )

    if aux_init_at_prev
        @inbounds tmp = deepcopy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_log_lik_ratio = -γ_curr * (
        sum(x -> dist(x, S_curr) for x in  aux_data)
        - sum(x -> dist(x, S_prop) for x in aux_data)
    )

    suff_stat_prop = sum(x -> dist(x, S_prop) for x in data)
    log_lik_ratio = -γ_curr * (
        suff_stat_prop - suff_stat_curr
    )

    log_multinom_ratio_term = log_multinomial_ratio(S_curr, S_prop)

    log_prior_ratio = -γ_prior * (
        dist(S_prop, mode_prior) - dist(S_curr, mode_prior)
    )

    # Log acceptance probability
    log_α = (
        log_lik_ratio + 
        log_prior_ratio + 
        aux_log_lik_ratio + 
        log_ratio +
        log_multinom_ratio_term
    )

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        copy!(S_curr[i], I_tmp)
        return 1, suff_stat_prop
    else 
        copy!(I_tmp, S_curr[i])
        return 0, suff_stat_curr
    end 

end 

function imcmc_gibbs_scan(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence,
    γ_curr::Float64,
    posterior::SIM, 
    mcmc::SimMcmcInsertDeleteGibbs,
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool, 
    
    )

    β = mcmc.β




end