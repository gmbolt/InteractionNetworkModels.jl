
function iex_gibbs_insert_delete_update!(
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


function iex_gibbs_flip_update!(
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

function iex_gibbs_scan!(
    S_curr::InteractionSequence,
    S_prop::InteractionSequence,
    γ_curr::Float64,
    posterior::SIM, 
    mcmc::SimMcmcInsertDeleteGibbs,
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool, 
    count::Vector{Int},
    acc_count::Vector{Int}
    )

    for i in 1:length(S_curr)
        if rand() < 0.5 
            was_acc, suff_stat_curr = iex_gibbs_insert_delete_update!(
                S_curr, S_prop, 
                γ_curr, i, 
                posterior, mcmc,
                P, aux_data, 
                suff_stat_curr, 
                aux_init_at_prev
            )
            count[1] += 1 
            acc_count[1] += was_acc
        else
            was_acc, suff_stat_curr = iex_gibbs_insert_delete_update!(
                S_curr, S_prop, 
                γ_curr, i, 
                posterior, mcmc,
                P, aux_data, 
                suff_stat_curr, 
                aux_init_at_prev
            )
            count[2] += 1 
            acc_count[2] += was_acc
        end 
    end
end


function draw_sample_mode!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::Union{SimIexInsertDelete,SimIexInsertDeleteProportional},
    posterior::SimPosterior,
    γ_fixed::Float64;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) 

    if loading_bar
        iter = Progress(
            length(sample_out) * lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 

    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    β = mcmc.β
    aux_mcmc = mcmc.aux_mcmc

    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for i in 1:length(S_init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], S_init[i])
        copy!(S_prop[i], S_init[i])
    end 

    γ_curr = γ_fixed

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 0 # Keeps track all samples (included lags and burn_ins) 

    tr_dim_count = 0 
    tr_dim_acc_count = 0
    ed_count = 0 
    ed_acc_count = 0
    flp_count = 0
    flp_acc_count = 0

    aux_data = [[Int[]] for i in 1:posterior.sample_size]
    # Initialise the aux_data 
    aux_model = SIM(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer)
    draw_sample!(aux_data, aux_mcmc, aux_model, burn_in=10000)

    # Evaluate sufficient statistic
    suff_stat_curr = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
    )
    suff_stats = Float64[suff_stat_curr] # Storage for all sufficient stats (for diagnostics)
    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)
    p_ins = get_informed_insertion_dist(posterior, mcmc.α)
    while sample_count ≤ length(sample_out)
        i += 1
        # Store value 
        if (i > burn_in) & (((i - burn_in - 1) % lag)==0)
            @inbounds sample_out[sample_count] = deepcopy(S_curr)
            push!(suff_stats, suff_stat_curr)
            sample_count += 1
        end 
        # W.P. do update move (accept-reject done internally by function call)
        if rand() < β
            if rand() < 0.5 
                was_acc, suff_stat_curr = double_iex_multinomial_edit_accept_reject!(
                    S_curr, S_prop, 
                    posterior, γ_curr,
                    mcmc, P, 
                    aux_data,
                    suff_stat_curr,
                    aux_init_at_prev
                )
                ed_acc_count += was_acc
                ed_count += 1
            else 
                was_acc, suff_stat_curr = double_iex_flip_accept_reject!(
                    S_curr, S_prop, 
                    posterior, γ_curr,
                    mcmc, P, 
                    aux_data,
                    suff_stat_curr, 
                    aux_init_at_prev
                )
                flp_acc_count += was_acc
                flp_count += 1
            end 
        # Else do trans-dim move. We will do accept-reject move here 
        else 
            was_acc, suff_stat_curr = double_iex_trans_dim_informed_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr,
                mcmc,
                p_ins,
                aux_data,
                suff_stat_curr,
                aux_init_at_prev
            )
            tr_dim_acc_count += was_acc
            tr_dim_count += 1
        end 
        if loading_bar
            next!(iter)
        end 
    end 
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return (
        ed_count, ed_acc_count,
        flp_count, flp_acc_count,
        tr_dim_count, tr_dim_acc_count,
        suff_stats
    )
end 


function accept_reject_mode!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior,
    γ_curr::Float64, 
    mcmc::Union{SimIexInsertDelete,SimIexInsertDeleteProportional},
    P::CumCondProbMatrix,
    p_ins::Categorical,
    aux_data::InteractionSequenceSample{Int},
    acc_count::Vector{Int},
    count::Vector{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool=false
    ) 
    
    β = mcmc.β
    if rand() < β
        if rand() < 0.5 
            was_accepted, suff_stat_curr = double_iex_multinomial_edit_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr, 
                mcmc, P, 
                aux_data,
                suff_stat_curr,
                aux_init_at_prev
            )
            acc_count[1] += was_accepted
            count[1] += 1
        else 
            was_accepted, suff_stat_curr = double_iex_flip_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr, 
                mcmc, P, 
                aux_data,
                suff_stat_curr,
                aux_init_at_prev
            )
            acc_count[2] += was_accepted
            count[2] += 1
        end 
    else 
        was_accepted, suff_stat_curr = double_iex_trans_dim_informed_accept_reject!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc, p_ins,
            aux_data,
            suff_stat_curr,
            aux_init_at_prev
        )
        acc_count[3] += was_accepted 
        count[3] += 1
    end 
    return suff_stat_curr
end 

