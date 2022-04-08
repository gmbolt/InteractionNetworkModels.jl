using IterTools, StatsBase
export imcmc_multi_split_involution!, imcmc_multi_merge_involution!
export imcmc_multi_split_prop_sample!, imcmc_multi_merge_prop_sample!
export imcmc_special_multi_split_prop_sample!, imcmc_special_multi_merge_prop_sample!
export log_binomial_ratio_split_merge

function imcmc_multi_split_involution!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    ind_split::Vector{Int},
    ind_loc::Vector{Int},
    k_vec::Vector{Int}, # Split locations (in each path),
    curr_pointers::InteractionSequence{T},
    prop_pointers::InteractionSequence{T}
    ) where {T}

    store = Vector{T}[]
    log_ratio_inc = 0.0 # Increment to log_ratio 
    for (i,k) in Iterators.reverse(zip(ind_split, k_vec))
        tmp1 = popat!(S_prop, i)
        n = length(tmp1)
        @assert n>1 "Cannot split path of length one."
        log_ratio_inc += log(n-1)
        tmp2 = popfirst!(prop_pointers)
        copy!(tmp2, tmp1[(k+1):end])
        resize!(tmp1, k)
        pushfirst!(store, tmp2)
        pushfirst!(store, tmp1)
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    return log_ratio_inc

end 

function imcmc_multi_split_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T
    ) where {T<:SimPosteriorSampler}

    ν = mcmc.ν_td
    N = length(S_curr)
    ε = rand(1:min(N, ν))
    ind_split = view(mcmc.ind_td_split, 1:ε)
    ind_loc = view(mcmc.ind_td_merge, 1:2ε)
    prop_pointers = mcmc.curr_pointers

    store = mcmc.store 
    empty!(store)

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_split)
    StatsBase.seqsample_a!(1:(N+ε), ind_loc)

    for i in Iterators.reverse(ind_split)
        @assert length(S_prop[i])>1 "Cannot split path of length one."
        tmp1 = popat!(S_prop, i)
        n = length(tmp1)
        k = rand(1:(n-1)) # Split location 
        log_ratio += log(n-1)
        tmp2 = popfirst!(prop_pointers)
        copy!(tmp2, tmp1[(k+1):end])
        resize!(tmp1, k)
        pushfirst!(store, tmp2)
        pushfirst!(store, tmp1)
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(N, ν)) - log(min(floor(length(S_prop/2)), ν))

    return log_ratio 

end 


function imcmc_multi_merge_involution!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    ind_merge::Vector{Int},
    ind_loc::Vector{Int},
    curr_pointers::InteractionSequence{T},
    prop_pointers::InteractionSequence{T}
    ) where {T}

    store = Vector{T}[]
    log_ratio_inc = 0.0 
    i = length(ind_merge)
    while i > 0
        # Merge i-1 and i 
        tmp2 = popat!(S_prop, ind_merge[i])
        tmp1 = popat!(S_prop, ind_merge[i-1])
        append!(tmp1, tmp2)
        pushfirst!(prop_pointers, tmp2)
        pushfirst!(store, tmp1)
        log_ratio_inc -= log(length(tmp1)-1)
        i -= 2 
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    return log_ratio_inc
end 

function imcmc_multi_merge_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T
    ) where {T<:SimPosteriorSampler}

    ν = mcmc.ν_td 
    N = length(S_curr)
    ε = rand(1:min(floor(Int, N/2), ν))

    ind_merge = view(mcmc.ind_td_merge, 1:2ε)
    ind_loc = view(mcmc.ind_td_split, 1:ε)

    store = mcmc.store 
    empty!(store)
    prop_pointers = mcmc.prop_pointers

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_merge)
    StatsBase.seqsample_a!(1:(N-ε), ind_loc)

    i = length(ind_merge)
    while i > 0
        # Merge i-1 and i 
        tmp2 = popat!(S_prop, ind_merge[i])
        tmp1 = popat!(S_prop, ind_merge[i-1])
        append!(tmp1, tmp2)
        pushfirst!(prop_pointers, tmp2)
        pushfirst!(store, tmp1)
        log_ratio -= log(length(tmp1)-1)
        i -= 2 
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(floor(Int, N/2), ν)) - log(min(length(S_prop), ν))

    return log_ratio 
end 

function log_binomial_ratio_split_merge(N, τ, ε)
    # \binom{N-τ}{ε}/\binom{N}{ε}
    out = 0.0
    if τ < ε 
        for j in 0:(τ-1)
            out += log(N-ε-j) - log(N-j)
        end 
    else 
        for j in 0:(ε-1)
            out += log(N-τ-j) - log(N-j)
        end 
    end
    return out 
end 

# Special case (when we have paths of length one present)
function imcmc_special_multi_split_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T
    ) where {T<:SimPosteriorSampler}

    ν = mcmc.ν_td
    N = length(S_curr)
    ind_ok_len = findall(x->length(x)>1, S_curr) # Good paths (length > 1)
    τ = N-length(ind_ok_len)                     # Number of bad paths 
    ε = rand(1:min(N-τ, ν))     
    ind_split = view(mcmc.ind_td_split, 1:ε)
    ind_loc = view(mcmc.ind_td_merge, 1:2ε)
    prop_pointers = mcmc.curr_pointers

    store = mcmc.store 
    empty!(store)

    log_ratio = 0.0

    StatsBase.seqsample_a!(ind_ok_len, ind_split)  # Sample subsequence of good paths
    StatsBase.seqsample_a!(1:(N+ε), ind_loc)

    for i in Iterators.reverse(ind_split)
        @assert length(S_prop[i])>1 "Cannot split path of length one."
        tmp1 = popat!(S_prop, i)
        n = length(tmp1)
        k = rand(1:(n-1)) # Split location 
        log_ratio += log(n-1)
        tmp2 = popfirst!(prop_pointers)
        copy!(tmp2, tmp1[(k+1):end])
        resize!(tmp1, k)
        pushfirst!(store, tmp2)
        pushfirst!(store, tmp1)
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(N-τ, ν)) - log(min(floor(length(S_prop/2)), ν))

    log_ratio += log_binomial_ratio_split_merge(N, τ, ε)

    return log_ratio 

end 



function imcmc_special_multi_merge_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T
    ) where {T<:SimPosteriorSampler}

    ν = mcmc.ν_td 
    N = length(S_curr)
    ε = rand(1:min(floor(Int, N/2), ν))

    ind_merge = view(mcmc.ind_td_merge, 1:2ε)
    ind_loc = view(mcmc.ind_td_split, 1:ε)

    store = mcmc.store 
    empty!(store)
    prop_pointers = mcmc.prop_pointers

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_merge)
    StatsBase.seqsample_a!(1:(N-ε), ind_loc)

    i = length(ind_merge)
    while i > 0
        # Merge i-1 and i 
        tmp2 = popat!(S_prop, ind_merge[i])
        tmp1 = popat!(S_prop, ind_merge[i-1])
        append!(tmp1, tmp2)
        pushfirst!(prop_pointers, tmp2)
        pushfirst!(store, tmp1)
        log_ratio -= log(length(tmp1)-1)
        i -= 2 
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    M = length(S_prop)
    τ = sum(length(x)==1 for x in S_prop) # Find number of length one paths
    log_ratio -= log_binomial_ratio_split_merge(M, τ, ε)

    log_ratio += log(min(floor(Int, N/2), ν)) - log(min(M-τ, ν))

    return log_ratio 
end 

# Adding necessary functions for mcmc samplers to work 
# We first need the accept reject function for this move 
function double_iex_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior, 
    γ_curr::Float64,
    mcmc::SimIexSplitMerge,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    )  
    
    K_inner, K_outer = (posterior.K_inner, posterior.K_outer)
    K_out_lb, K_out_ub = (K_outer.l, K_outer.u)
    K_in_lb, K_in_ub = (K_inner.l, K_inner.u)
    data = posterior.data 
    dist = posterior.dist 
    V = posterior.V 
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode

    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    aux_mcmc = mcmc.aux_mcmc 

    log_ratio = 0.0

    do_split = false

    τ = sum(length(x)==1 for x in S_curr)
    N = length(S_curr)
    # If current state has one path of length one, nothing can be done
    if (N==τ==1)
        return 0, suff_stat_curr
    # If one path of length > 1 we split with probability one 
    elseif N==1
        do_split = true 
        log_ratio += imcmc_multi_split_prop_sample!(S_curr, S_prop, mcmc)
        log_ratio += all(length(x)==1 for x in S_prop) ? 0.0 : log(0.5)
    # If no length one paths, use std proposal fn 
    elseif τ==0 
        do_split = rand(Bool)
        if do_split
            log_ratio += imcmc_multi_split_prop_sample!(S_curr, S_prop, mcmc)
        else 
            log_ratio += imcmc_multi_merge_prop_sample!(S_curr, S_prop, mcmc)
        end 
    # If all paths are length one, do merge with probability one
    elseif τ==N
        do_split = false
        log_ratio += imcmc_special_multi_merge_prop_sample!(S_curr, S_prop, mcmc)
        log_ratio += log(0.5) # Accounts for uneven choice between this and split move
    # Else to use special proposal function 
    else 
        do_split = rand(Bool)
        if do_split
            log_ratio += imcmc_special_multi_split_prop_sample!(S_curr, S_prop, mcmc)
        else 
            log_ratio += imcmc_special_multi_merge_prop_sample!(S_curr, S_prop, mcmc)
        end 
    end 

    # Account for dimension bounds (make acc_prob 0 if outside bounds)
    if !(K_out_lb ≤ length(S_prop) ≤ K_out_ub)
        log_ratio += -Inf
    end 
    if any(!(K_in_lb ≤ length(x) ≤ K_in_ub) for x in S_prop)
        log_ratio += -Inf
    end 

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

    ε = abs(length(S_curr)-length(S_prop))
    if log(rand()) < log_α
        # We accept!
        if do_split
            # We've increased dimension
            # Bring in pointers from curr_pointers 
            for i in 1:ε
                tmp = pop!(curr_pointers)
                push!(S_curr,tmp)
            end 
        else 
            # We've decreased dimension
            # Send pointers back to curr_pointers 
            for i in 1:ε
                tmp = pop!(S_curr)
                push!(curr_pointers, tmp)
            end 
        end 
        # Copy across from S_prop 
        for (x,y) in zip(S_curr,S_prop)
            copy!(x, y)
        end 
        return 1, suff_stat_prop
    else 
        # We reject!
        if do_split
            # We've increased dimension
            # Send back pointers to prop_pointers 
            for i in 1:ε
                tmp = pop!(S_prop)
                push!(prop_pointers, tmp)
            end 
        else 
            # We've decreased dimension
            # Bring in pointers from prop_pointers
            for i in 1:ε
                tmp = pop!(prop_pointers)
                push!(S_prop, tmp)
            end 
        end 
        # Copy across from S_curr
        for (x,y) in zip(S_prop,S_curr)
            copy!(x, y)
        end 
        return 0, suff_stat_curr
    end 

end 

function draw_sample_mode!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SimIexSplitMerge,
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
            was_acc, suff_stat_curr = double_iex_trans_dim_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr,
                mcmc,
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



# For Joint Distribution
# ----------------------

# Mode accept-reject 

function accept_reject_mode!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SimPosterior,
    γ_curr::Float64, 
    mcmc::SimIexSplitMerge,
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
        was_accepted, suff_stat_curr = double_iex_trans_dim_accept_reject!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc,
            aux_data,
            suff_stat_curr,
            aux_init_at_prev
        )
        acc_count[3] += was_accepted 
        count[3] += 1
    end 
    return suff_stat_curr
end 