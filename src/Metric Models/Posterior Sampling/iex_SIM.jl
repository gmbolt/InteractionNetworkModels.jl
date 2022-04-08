using Distributions, StatsBase, ProgressMeter

export draw_sample_mode!, draw_sample_mode, draw_sample_gamma!, draw_sample_gamma
export draw_sample!, draw_sample


function double_iex_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior,
    γ_curr::Float64,
    mcmc::Union{SimIexInsertDelete,SimIexSplitMerge},
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    ) 

    N = length(S_curr)  
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
    log_ratio = 0.0

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
            d = rand(0:min(n-K_in_lb, δ_tmp))
            m = n + δ_tmp - 2*d

            # Catch invalid proposals
            if (m > K_in_ub)
                # Here we just reject the proposal
                for i in 1:N
                    @inbounds copy!(S_prop[i], S_curr[i])
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
                S_prop[i],
                ind_del, ind_add, vals_del, 
                P)

            @inbounds mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_ratio += log(min(n-K_in_lb, δ_tmp)+1) - log(min(m-K_in_lb, δ_tmp)+1)

        end 

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 
    
    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        tmp = deepcopy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_log_lik_ratio = γ_curr * (
        sum(x -> dist(x, S_prop), aux_data)    
        - sum(x -> dist(x, S_curr), aux_data)
    )

    suff_stat_prop = sum(x -> dist(x, S_prop), data)
    log_lik_ratio = γ_curr * (
        suff_stat_curr - suff_stat_prop
    )

    log_prior_ratio = -γ_prior * (
        dist(S_prop, mode_prior) - dist(S_curr, mode_prior)
    )

    log_multinom_term = log_multinomial_ratio(S_curr, S_prop)

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio + log_multinom_term
    # println("Edit alloc proposing...")
    # println("$(S_prop)")

    # @show log_lik_ratio, aux_log_lik_ratio, log_multinom_term, log_ratio, log_α
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

function double_iex_flip_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior,
    γ_curr::Float64,
    mcmc::Union{SimIexInsertDelete,SimIexInsertDeleteProportional,SimIexSplitMerge},
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64, 
    aux_init_at_prev::Bool
    ) 
    
    dist = posterior.dist
    V = posterior.V
    data = posterior.data
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode
    K_inner, K_outer = (posterior.K_inner, posterior.K_outer)

    aux_mcmc = mcmc.aux_mcmc
    lengths = length.(S_curr)

    δ = rand(1:min(mcmc.ν_ed,sum(lengths)))
    # @show δ, lengths
    alloc = rand_restr_bins(lengths, δ)
    ind = mcmc.ind_add
    log_ratio = 0.0

    for (key,val) in pairs(alloc)
        ind_flip = view(ind, 1:val)
        @inbounds StatsBase.seqsample_a!(1:lengths[key], ind_flip)
        log_ratio += flip_informed_excl!(
            S_prop[key], 
            ind_flip, 
            P
        ) 
    end 
    
    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        tmp = deepcopy(aux_data[end])
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

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in keys(alloc)
            @inbounds copy!(S_curr[i], S_prop[i])
        end
        return 1, suff_stat_prop
    else 
        for i in keys(alloc)
            @inbounds copy!(S_prop[i], S_curr[i])
        end 
        return 0, suff_stat_curr
    end 

end 


function double_iex_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior, 
    γ_curr::Float64,
    mcmc::Union{SimIexInsertDelete,SimIexInsertDeleteProportional},
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    )  
    
    K_inner, K_outer = (posterior.K_inner, posterior.K_outer)
    K_out_lb, K_out_ub = (K_outer.l, K_outer.u)
    K_in_ub = K_inner.u
    data = posterior.data 
    dist = posterior.dist 
    V = posterior.V 
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode


    ν_td = mcmc.ν_td
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    aux_mcmc = mcmc.aux_mcmc 

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    ε = rand(1:ν_td)
    d = rand(0:min(ν_td, N))
    a = ε - d
    # Catch invalid proposal (outside dimension bounds)
    M = N - d + a  # Number of paths in proposal
    if (M < K_out_lb) | (M > K_out_ub)
        return 0, suff_stat_curr
    end 
    if d > 0 
        log_ratio += imcmc_multi_delete_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            d, 
            V
        )
    end 
    if a > 0 
        log_ratio += imcmc_multi_insert_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            a, 
            V, K_in_ub
        )
    end 

    log_ratio += log(min(ν_td, N) + 1) - log(min(ν_td, M) + 1)

    # Now do accept-reject step (**THIS IS WHERE WE DIFFER FROM MODEL SAMPLER***)
    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        tmp = deepcopy(aux_data[end])
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

    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
    if log(rand()) < log_α
        # Do deletions to S_curr
        ind = view(mcmc.ind_td_del, 1:d)
        for i in Iterators.reverse(ind) # If we didnt do reverse would have to update indices 
            @inbounds tmp = popat!(S_curr, i)
            pushfirst!(curr_pointers, tmp)
        end
        # Do insertions 
        ind = view(mcmc.ind_td_add, 1:a)
        for i in ind 
            tmp = popfirst!(curr_pointers)
            copy!(tmp, S_prop[i])
            insert!(S_curr, i, tmp)
        end 
        return 1, suff_stat_prop
    else 
        # Remove insertions and return to prop_pointers 
        ind = view(mcmc.ind_td_add, 1:a)
        for i in Iterators.reverse(ind)
            @inbounds tmp = popat!(S_prop, i)
            pushfirst!(prop_pointers, tmp)
        end 
        # Re-insert deletions 
        ind = view(mcmc.ind_td_del, 1:d)
        for i in ind 
            @inbounds tmp = popfirst!(prop_pointers)
            copy!(tmp, S_curr[i])
            insert!(S_prop, i, tmp)
        end
        return 0, suff_stat_curr
    end 

end 

function double_iex_trans_dim_informed_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior, 
    γ_curr::Float64,
    mcmc::T,
    p_ins::Categorical,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    ) where {T<:SimPosteriorSampler}
    
    K_inner, K_outer = (posterior.K_inner, posterior.K_outer)
    K_out_lb, K_out_ub = (K_outer.l, K_outer.u)
    K_in_ub = K_inner.u
    data = posterior.data 
    dist = posterior.dist 
    V = posterior.V 
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode


    ν_td = mcmc.ν_td
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    aux_mcmc = mcmc.aux_mcmc 

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    ε = rand(1:ν_td)
    d = rand(0:min(ν_td, N))
    a = ε - d
    # Catch invalid proposal (outside dimension bounds)
    M = N - d + a  # Number of paths in proposal
    if (M < K_out_lb) | (M > K_out_ub)
        return 0, suff_stat_curr
    end 
    if d > 0 
        log_ratio += imcmc_multi_delete_prop_sample_informed!(
            S_curr, S_prop, 
            mcmc, 
            d, 
            p_ins
        )
    end 
    if a > 0 
        log_ratio += imcmc_multi_insert_prop_sample_informed!(
            S_curr, S_prop, 
            mcmc, 
            a, 
            p_ins,
            K_in_ub
        )
    end 

    log_ratio += log(min(ν_td, N) + 1) - log(min(ν_td, M) + 1)
    # Now do accept-reject step (**THIS IS WHERE WE DIFFER FROM MODEL SAMPLER***)
    aux_model = SIM(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    if aux_init_at_prev
        tmp = deepcopy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_suff_stat_curr = mapreduce(x -> dist(x, S_curr), + , aux_data)
    aux_suff_stat_prop = mapreduce(x -> dist(x, S_prop), +, aux_data)
    aux_log_lik_ratio = -γ_curr * (
        aux_suff_stat_curr - aux_suff_stat_prop
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

    # println("Acceptance prob: $(exp(log_α))") 
    # @show is_insert, suff_stat_curr, suff_stat_prop, log_lik_ratio, aux_suff_stat_curr, aux_suff_stat_prop, aux_log_lik_ratio, log_ratio, log_multinom_term, log_prior_ratio, log_α, exp(log_α)
    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
    if log(rand()) < log_α
        # Do deletions to S_curr
        ind = view(mcmc.ind_td_del, 1:d)
        for i in Iterators.reverse(ind) # If we didnt do reverse would have to update indices 
            @inbounds tmp = popat!(S_curr, i)
            pushfirst!(curr_pointers, tmp)
        end
        # Do insertions 
        ind = view(mcmc.ind_td_add, 1:a)
        for i in ind 
            tmp = popfirst!(curr_pointers)
            copy!(tmp, S_prop[i])
            insert!(S_curr, i, tmp)
        end 
        return 1, suff_stat_prop
    else 
        # Remove insertions and return to prop_pointers 
        ind = view(mcmc.ind_td_add, 1:a)
        for i in Iterators.reverse(ind)
            @inbounds tmp = popat!(S_prop, i)
            pushfirst!(prop_pointers, tmp)
        end 
        # Re-insert deletions 
        ind = view(mcmc.ind_td_del, 1:d)
        for i in ind 
            @inbounds tmp = popfirst!(prop_pointers)
            copy!(tmp, S_curr[i])
            insert!(S_prop, i, tmp)
        end
        return 0, suff_stat_curr
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

function draw_sample_mode(
    mcmc::T,
    posterior::SimPosterior,
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) where {T<:SimPosteriorSampler}

    sample_out = Vector{InteractionSequence{Int}}(undef, desired_samples)
    draw_sample_mode!(
        sample_out, 
        mcmc, posterior, 
        γ_fixed, 
        burn_in=burn_in, lag=lag, S_init=S_init,
        loading_bar=loading_bar,
        aux_init_at_prev=aux_init_at_prev
        )
    return sample_out

end 

function (mcmc::T where {T<:SimPosteriorSampler})(
    posterior::SimPosterior, 
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    )
    sample_out = Vector{InteractionSequence{Int}}(undef, desired_samples)

    (
        edit_count, edit_acc_count, 
        flip_count, flip_acc_count,
        trans_dim_count, trans_dim_acc_count,
        suff_stats
        ) = draw_sample_mode!(
            sample_out, 
            mcmc, 
            posterior, γ_fixed, 
            burn_in=burn_in, 
            lag=lag, 
            S_init=S_init,
            loading_bar=loading_bar,
            aux_init_at_prev=aux_init_at_prev
            )

    p_measures = Dict(
        "Proportion Update Moves" => (edit_count+flip_count)/(edit_count+flip_count+trans_dim_count),
        "Edit Alloc Move Acceptance Probability" => edit_acc_count / edit_count,
        "Flip Alloc Acceptance Probability" => flip_acc_count / flip_count,
        "Trans-Dimensional Move Acceptance Probability" => trans_dim_acc_count / trans_dim_count
        )
    output = SimPosteriorModeConditionalMcmcOutput(
            γ_fixed, 
            sample_out, 
            posterior,
            suff_stats,
            p_measures
            )

    return output

end 


# Dispersion Conditional 
# ----------------------

function draw_sample_gamma!(
    sample_out::Union{Vector{Float64}, SubArray},
    mcmc::T,
    posterior::SimPosterior,
    S_fixed::InteractionSequence{Int};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64=4.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) where {T<:SimPosteriorSampler}

    if loading_bar
        iter = Progress(
            length(sample_out) * lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (dispersion conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 

    # Define aliases for pointers to the storage of current vals and proposals
    ε = mcmc.ε
    aux_mcmc = mcmc.aux_mcmc

    acc_count = 0
    i = 1 # Which iteration we are on 
    sample_count = 1  # Which sample we are working to get 

    S_curr = deepcopy(S_fixed)
    γ_curr = γ_init
    aux_data = [[Int[]] for i in 1:posterior.sample_size]

    # Evaluate sufficient statistic
    suff_stat = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
        )

    # Initialise the aux_data 
    aux_model = SIM(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer
    )
    
    draw_sample!(aux_data, aux_mcmc, aux_model, burn_in=10000)

    while sample_count ≤ length(sample_out)
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out[sample_count] = γ_curr
            sample_count += 1
        end 

        γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

        aux_model = SIM(
            S_curr, γ_prop, 
            posterior.dist, 
            posterior.V, 
            posterior.K_inner, posterior.K_outer
        )
        if aux_init_at_prev
            tmp = deepcopy(aux_data[end])
            draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
        else 
            draw_sample!(aux_data, aux_mcmc, aux_model)
        end 


        # Accept reject

        log_lik_ratio = (γ_curr - γ_prop) * suff_stat
        aux_log_lik_ratio = (γ_prop - γ_curr) * sum_of_dists(aux_data, S_curr, posterior.dist)

        log_α = (
            logpdf(posterior.γ_prior, γ_prop) 
            - logpdf(posterior.γ_prior, γ_curr)
            + log_lik_ratio + aux_log_lik_ratio 
        )
        if log(rand()) < log_α
            γ_curr = γ_prop
            acc_count += 1
        end 
        if loading_bar
            next!(iter)
        end 
        i += 1

    end 
    return acc_count
end 

function draw_sample_gamma(
    mcmc::T,
    posterior::SimPosterior,
    S_fixed::InteractionSequence{Int};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) where {T<:SimPosteriorSampler}

    sample_out = Vector{Float64}(undef, desired_samples)
    draw_sample_gamme!(
        sample_out, 
        mcmc, posterior, 
        S_fixed, 
        burn_in=burn_in, lag=lag, γ_init=γ_init,
        loading_bar=loading_bar,
        aux_init_at_prev=aux_init_at_prev
        )
    return sample_out

end 


function (mcmc::T where {T<:SimPosteriorSampler})(
    posterior::SimPosterior, 
    S_fixed::InteractionSequence{Int};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) 

    sample_out = Vector{Float64}(undef, desired_samples)

    
    acc_count = draw_sample_gamma!(
            sample_out, 
            mcmc, 
            posterior, S_fixed, 
            burn_in=burn_in, 
            lag=lag, 
            γ_init=γ_init,
            loading_bar=loading_bar,
            aux_init_at_prev=aux_init_at_prev
            )

    p_measures = Dict(
            "Acceptance Probability" => acc_count/desired_samples
        )

    output = SimPosteriorDispersionConditionalMcmcOutput(
            S_fixed, 
            sample_out, 
            posterior.γ_prior,
            posterior.data,
            p_measures
            )

    return output

end

# Joint Distribution 
# ------------------

# Mode accept-reject 

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

# Gamma accept-reject

function accept_reject_gamma!(
    γ_curr::Float64,
    S_curr::InteractionSequence{Int},
    posterior::SimPosterior,
    mcmc::T,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool=false
    ) where {T<:SimPosteriorSampler}

    ε = mcmc.ε
    aux_mcmc = mcmc.aux_mcmc    
    dist = posterior.dist

    γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

    aux_model = SIM(
        S_curr, γ_prop, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, posterior.K_outer
    )
    if aux_init_at_prev
        @inbounds tmp = deepcopy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 
    # Accept reject

    log_lik_ratio = (γ_curr - γ_prop) * suff_stat_curr

    aux_log_lik_ratio = (γ_prop - γ_curr) * sum(dist(x,S_curr) for x in aux_data)

    log_α = (
        logpdf(posterior.γ_prior, γ_prop) 
        - logpdf(posterior.γ_prior, γ_curr)
        + log_lik_ratio + aux_log_lik_ratio 
    )
    if log(rand()) < log_α
        return γ_prop, 1
    else 
        return γ_curr, 0
    end 
end 

function draw_sample!(
    sample_out_S::Union{InteractionSequenceSample{Int},SubArray},
    sample_out_gamma::Union{Vector{Float64},SubArray},
    mcmc::T,
    posterior::SimPosterior;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) where {T<:SimPosteriorSampler}

    if loading_bar
        iter = Progress(
            length(sample_out_S) * lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (joint)....")  # Loading bar. Minimum update interval: 1 second
    end 

    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    aux_mcmc = mcmc.aux_mcmc

    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for i in 1:length(S_init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], S_init[i])
        copy!(S_prop[i], S_init[i])
    end 
    γ_curr = copy(γ_init)

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 1 # Keeps track all samples (included lags and burn_ins) 

    acc_count = [0,0,0]
    count = [0,0,0]
    γ_acc_count = 0

    aux_data = [[Int[]] for i in 1:posterior.sample_size]
    # Initialise the aux_data 
    aux_model = SIM(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer)
    draw_sample!(aux_data, aux_mcmc, aux_model, burn_in=10000)
    # Initialise sufficient statistic
    suff_stat_curr = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
    )
    suff_stats = Float64[suff_stat_curr]
    # Get informed proposal matrix
    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)
    p_ins = get_informed_insertion_dist(posterior, mcmc.α)
    while sample_count ≤ length(sample_out_S)
        # Store values
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out_S[sample_count] = deepcopy(S_curr)
            @inbounds sample_out_gamma[sample_count] = copy(γ_curr)
            push!(suff_stats, suff_stat_curr)
            sample_count += 1
        end 

        # Update mode
        # ----------- 
        suff_stat_curr = accept_reject_mode!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc, P, p_ins,
            aux_data, 
            acc_count, count,
            suff_stat_curr,
            aux_init_at_prev
        )
        # Update gamma 
        # ------------
        γ_curr, tmp =  accept_reject_gamma!(
            γ_curr,
            S_curr,
            posterior, 
            mcmc, 
            aux_data,
            suff_stat_curr, 
            aux_init_at_prev
        )
        γ_acc_count += tmp
        if loading_bar 
            next!(iter)
        end 
        i += 1
    end 

    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 

    ed_acc_prob = acc_count[1] / count[1]
    flip_acc_prob = acc_count[2] / count[2]
    td_acc_prob = acc_count[3] / count[3]
    γ_acc_prob = γ_acc_count / sum(count)
    return ed_acc_prob, flip_acc_prob, td_acc_prob, γ_acc_prob, suff_stats
end 

function draw_sample(
    mcmc::Union{SimIexInsertDelete,SimIexInsertDeleteProportional},
    posterior::SimPosterior;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) 

    sample_out_S = Vector{InteractionSequence{Int}}(undef, desired_samples)
    sample_out_gamma = Vector{Float64}(undef, desired_samples)

    draw_sample!(
        sample_out_S,
        sample_out_gamma, 
        mcmc, 
        posterior, 
        burn_in=burn_in, lag=lag, 
        S_init=S_init, γ_init=γ_init,
        loading_bar=loading_bar,
        aux_init_at_prev=aux_init_at_prev
    )

    return (S=sample_out_S, gamma=sample_out_gamma)
end 

function (mcmc::T where {T<:SimPosteriorSampler})(
    posterior::SimPosterior;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) 

    sample_out_S = Vector{InteractionSequence{Int}}(undef, desired_samples)
    sample_out_gamma = Vector{Float64}(undef, desired_samples)

    ed_acc_prob, flip_acc_prob, td_acc_prob, γ_acc_prob, suff_stats = draw_sample!(
        sample_out_S,
        sample_out_gamma, 
        mcmc, 
        posterior, 
        burn_in=burn_in, lag=lag, 
        S_init=S_init, γ_init=γ_init,
        loading_bar=loading_bar,
        aux_init_at_prev=aux_init_at_prev
    )

    p_measures = Dict(
        "Dipsersion acceptance probability" => γ_acc_prob,
        "Edit allocation acceptance probability" => ed_acc_prob,
        "Flip allocation acceptance probability" => flip_acc_prob,
        "Trans-dimensional acceptance probability" => td_acc_prob
        )

    return SimPosteriorMcmcOutput(
        sample_out_S, 
        sample_out_gamma, 
        posterior,
        suff_stats,
        p_measures
    )
    
end 