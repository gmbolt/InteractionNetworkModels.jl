using Distributions, StatsBase, ProgressMeter

export draw_sample_mode!, draw_sample_mode
export draw_sample_gamma!, draw_sample_gamma
export rand_multivariate_bernoulli


# NOTE μ_cusum must have 0.0 first entry, and for n bins μ_cusum must be of size n+1
function rand_multivariate_bernoulli(μ_cusum::Vector{Float64})
    @assert μ_cusum[1] ≈ 0 "First entry must be 0.0 (for differencing to find probabilities)."
    β = rand()
    for i in 1:length(μ_cusum)
        if β < μ_cusum[i]
            return i-1, μ_cusum[i]-μ_cusum[i-1]
        else 
            continue 
        end 
    end 
end 

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    mcmc::SisIexInsertDeleteEdit{T},
    ind::AbstractVector{T}
    ) where {T<:Union{Int, String}}

    prop_pointers = mcmc.prop_pointers
    ν_trans_dim = mcmc.ν_trans_dim
    N = length(S_curr)
    path_dist = mcmc.path_dist

    log_ratio = 0.0 
    for i in ind 
        migrate!(S_prop, prop_pointers, i, 1)
        rand!(S_prop[i], path_dist)
        log_ratio += - logpdf(path_dist, S_prop[i])
    end 
    log_ratio += log(ν_trans_dim) - log(min(ν_trans_dim,N)) 
    return log_ratio 

end 

function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T}, 
    mcmc::SisIexInsertDeleteEdit{T},
    ind::AbstractVector{T}
    ) where {T<:Union{Int,String}}

    prop_pointers = mcmc.prop_pointers
    ν_trans_dim = mcmc.ν_trans_dim
    N = length(S_curr)
    path_dist = mcmc.path_dist

    log_ratio = 0.0

    for i in Iterators.reverse(ind)
        migrate!(prop_pointers, S_prop, 1, i)
        log_ratio += logpdf(path_dist, S_curr[i])
    end 

    log_ratio += log(min(ν_trans_dim,N)) - log(ν_trans_dim)
    return log_ratio

end 

function delete_insert_informed!(
    x::Path, 
    ind_del::AbstractArray{Int}, 
    ind_add::AbstractArray{Int}, 
    vals_del::AbstractArray{Int},
    P::CumCondProbMatrix
    )

    @views for (i, index) in enumerate(ind_del)
        tmp_ind = index - i + 1  # Because size is changing must adapt index
        vals_del[i] = x[tmp_ind]
        deleteat!(x, tmp_ind)
    end 

    # Now find distrbution for insertions via co-occurence
    curr_vertices = unique(x)
    if length(curr_vertices) == 0 
        V = size(P)[2]
        tmp = fill(1/V, V)
        pushfirst!(tmp, 0.0)
        μ_cusum = cumsum(tmp)
    else 
        μ_cusum = sum(P[:,curr_vertices], dims=2)[:] ./ length(curr_vertices)
    end 

    # @show μ_cusum

    log_ratio_entries = 0.0
    # Add probability of values deleted
    for v in vals_del
        log_ratio_entries += log(μ_cusum[v+1]-μ_cusum[v])
    end 

    # Sample new entries and add probabilility
    @views for index in ind_add
        # @show i, index, val
        val, prob = rand_multivariate_bernoulli(μ_cusum)
        insert!(x, index, val)
        log_ratio_entries += -log(prob)
    end 

    return log_ratio_entries
end 

function double_iex_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    posterior::SisPosterior{T},
    γ_curr::Float64,
    mcmc::SisIexInsertDeleteEdit{T},
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{T},
    suff_stat_curr::Float64
    ) where {T<:Int}
    
    N = length(S_curr)  
    dist = posterior.dist
    V = posterior.V
    K_inner = posterior.K_inner
    K_outer = posterior.K_outer
    data = posterior.data
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode

    aux_mcmc = mcmc.aux_mcmc

    δ = rand(1:mcmc.ν_edit)  # Number of edits to enact 
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
            # a,b = (lb(n, δ_tmp, model), ub(n, δ_tmp))
            d = rand(0:min(n,δ_tmp))
            m = n + δ_tmp - 2*d

            # Catch invalid proposals
            if (m < 1) | (m > K_inner)
                # Here we just reject the proposal
                for i in 1:N
                    copy!(S_prop[i], S_curr[i])
                end 
                return 0, suff_stat_curr
            end 

            # tot_dels += d
            # println("       Deleting $d and adding $(δ_tmp-d)")
            ind_del = view(mcmc.ind_del, 1:d)
            ind_add = view(mcmc.ind_add, 1:(δ_tmp-d))
            vals_del = view(mcmc.vals, 1:d)

            # println("           ind_del: $ind_del ; ind_add: $ind_add")

            # Sample indexing info and new entries (all in-place)
            StatsBase.seqsample_a!(1:n, ind_del)
            StatsBase.seqsample_a!(1:m, ind_add)

            # *** HERE IS DIFFERENT FROM MODEL SAMPLER ***
            # The delete_insert_informed() function does the sampling + editing 
            log_ratio += delete_insert_informed!(
                S_prop[i],
                ind_del, ind_add, vals_del, 
                P)

            mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_ratio += log(min(n, δ_tmp)+1) - log(min(m, δ_tmp)+1)

        end 

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 
    
    aux_model = SIS(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
        )

    draw_sample!(aux_data, aux_mcmc, aux_model)

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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    # @show log_dim_diff, log_prod_term, log_lik_ratio, log_α, mean_len_curr, mean_len_prop

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_curr[i], S_prop[i])
        end
        return 1, suff_stat_prop
    else 
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_prop[i], S_curr[i])
        end 
        return 0, suff_stat_curr
    end 
end 


function double_iex_trans_dim_accept_reject!(
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    posterior::SisPosterior{T}, 
    γ_curr::Float64,
    mcmc::SisIexInsertDeleteEdit{T},
    aux_data::InteractionSequenceSample{T},
    suff_stat_curr::Float64
    )  where {T<:Union{Int, String}}
    
    K_inner = posterior.K_inner
    K_outer = posterior.K_outer
    data = posterior.data 
    dist = posterior.dist 
    V = posterior.V 
    γ_prior = posterior.S_prior.γ 
    mode_prior = posterior.S_prior.mode


    ν_trans_dim = mcmc.ν_trans_dim
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    aux_mcmc = mcmc.aux_mcmc 

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    is_insert = rand(Bernoulli(0.5))
    if is_insert
        ε = rand(1:ν_trans_dim) # How many to insert 
        # Catch invalid proposal (ones which have zero probability)
        if (N + ε) > K_outer
            # Make no changes and imediately reject  
            return 0  
        end 
        ind_tr_dim = view(mcmc.ind_trans_dim, 1:ε) # Storage for where to insert 
        StatsBase.seqsample_a!(1:(N+ε), ind_tr_dim) # Sample where to insert 
        log_ratio += imcmc_multi_insert_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            ind_tr_dim
            ) # Enact move and catch log ratio term 
    else 
        ε = rand(1:min(ν_trans_dim, N)) # How many to delete
        # Catch invalid proposal (would go to empty inter seq)
        if ε == N 
            return 0 
        end  
        ind_tr_dim = view(mcmc.ind_trans_dim, 1:ε) # Storage
        StatsBase.seqsample_a!(1:N, ind_tr_dim) # Sample which to delete 
        log_ratio += imcmc_multi_delete_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            ind_tr_dim
            ) # Enact move and catch log ratio 
    end 

    # Now do accept-reject step (**THIS IS WHERE WE DIFFER FROM MODEL SAMPLER***)
    aux_model = SIS(
        S_prop, γ_curr, 
        dist, 
        V, 
        K_inner, 
        K_outer
        )

    draw_sample!(aux_data, aux_mcmc, aux_model)

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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
    if log(rand()) < log_α
        if is_insert
            for i in ind_tr_dim
                migrate!(S_curr, curr_pointers, i, 1)
                copy!(S_curr[i], S_prop[i])
            end 
        else 
            for i in Iterators.reverse(ind_tr_dim)
            migrate!(curr_pointers , S_curr, 1, i)
            end 
        end 
        return 1, suff_stat_prop
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
                copy!(S_prop[i], S_curr[i])
            end 
        end 
        return 0, suff_stat_curr
    end 

end 

# =========================
# Samplers 
# =========================

# Mode conditional 
# ----------------

function draw_sample_mode!(
    sample_out::Union{InteractionSequenceSample{T}, SubArray},
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T},
    γ_fixed::Float64;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    if loading_bar
        iter = Progress(
            length(sample_out), # How many iters 
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
    for i in 1:length(init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], init[i])
        copy!(S_prop[i], init[i])
    end 

    γ_curr = γ_fixed

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 0 # Keeps track all samples (included lags and burn_ins) 

    tr_dim_count = 0 
    tr_dim_acc_count = 0
    upd_count = 0 
    upd_acc_count = 0

    aux_data = [[T[]] for i in 1:posterior.sample_size]
    # Initialise the aux_data 
    aux_model = SIS(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer)
    draw_sample!(aux_data, aux_mcmc, aux_model)
    
    # Evaluate sufficient statistic
    suff_stat_curr = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
        )

    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)
    while sample_count ≤ length(sample_out)
        i += 1
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end 
        # W.P. do update move (accept-reject done internally by function call)
        if rand() < β
            was_acc, suff_stat_curr = double_iex_multinomial_edit_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr,
                mcmc, P, 
                aux_data,
                suff_stat_curr
            )
            upd_acc_count += was_acc
            upd_count += 1
        # Else do trans-dim move. We will do accept-reject move here 
        else 
            was_acc, suff_stat_curr = double_iex_trans_dim_accept_reject!(
                S_curr, S_prop, 
                posterior, γ_curr,
                mcmc,
                aux_data,
                suff_stat_curr
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
                upd_count, upd_acc_count,
                tr_dim_count, tr_dim_acc_count
            )
end 

function draw_sample_mode(
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T},
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    draw_sample_mode!(
        sample_out, 
        mcmc, posterior, 
        γ_fixed, 
        burn_in=burn_in, lag=lag, init=init,
        loading_bar=loading_bar
        )
    return sample_out

end 

function (mcmc::SisIexInsertDeleteEdit{Int})(
    posterior::SisPosterior{T}, 
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}
    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)

    (
        update_count, update_acc_count, 
        trans_dim_count, trans_dim_acc_count
        ) = draw_sample_mode!(
            sample_out, 
            mcmc, 
            posterior, γ_fixed, 
            burn_in=burn_in, 
            lag=lag, 
            init=init,
            loading_bar=loading_bar
            )

    p_measures = Dict(
            "Proportion Update Moves" => update_count/(update_count+trans_dim_count),
            "Update Move Acceptance Probability" => update_acc_count / update_count,
            "Trans-Dimensional Move Acceptance Probability" => trans_dim_acc_count / trans_dim_count
        )
    output = SisPosteriorModeConditionalMcmcOutput(
            γ_fixed, 
            sample_out, 
            posterior.dist, 
            posterior.S_prior, 
            posterior.data,
            p_measures
            )

    return output

end 

# Dispersion Conditional 
# ----------------------

function draw_sample_gamma!(
    sample_out::Union{Vector{Float64}, SubArray},
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T},
    S_fixed::InteractionSequence{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Float64=4.0,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    if loading_bar
        iter = Progress(
            length(sample_out), # How many iters 
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
    γ_curr = init
    aux_data = [[T[]] for i in 1:posterior.sample_size]

    # Evaluate sufficient statistic
    suff_stat = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
        )

    # Initialise the aux_data 
    aux_model = SIS(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer)
    draw_sample!(aux_data, aux_mcmc, aux_model)

    while sample_count ≤ length(sample_out)
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = γ_curr
            sample_count += 1
        end 

        γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

        aux_model = SIS(
            S_curr, γ_prop, 
            posterior.dist, 
            posterior.V, 
            posterior.K_inner, posterior.K_outer
            )
        draw_sample!(aux_data, aux_mcmc, aux_model)

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
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T},
    S_fixed::InteractionSequence{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Float64,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Float64}(undef, desired_samples)
    draw_sample_gamme!(
        sample_out, 
        mcmc, posterior, 
        S_fixed, 
        burn_in=burn_in, lag=lag, init=init,
        loading_bar=loading_bar
        )
    return sample_out

end 


function (mcmc::SisIexInsertDeleteEdit{T})(
    posterior::SisPosterior{T}, 
    S_fixed::InteractionSequence{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Float64}(undef, desired_samples)

    
    acc_count = draw_sample_gamma!(
            sample_out, 
            mcmc, 
            posterior, S_fixed, 
            burn_in=burn_in, 
            lag=lag, 
            init=init,
            loading_bar=loading_bar
            )

    p_measures = Dict(
            "Acceptance Probability" => acc_count/desired_samples
        )

    output = SisPosteriorDispersionConditionalMcmcOutput(
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
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    posterior::SisPosterior{T},
    γ_curr::Float64, 
    mcmc::SisIexInsertDeleteEdit{T},
    P::CumCondProbMatrix,
    aux_data::InteractionSequenceSample{T},
    acc_count::Vector{Int},
    count::Vector{Int},
    suff_stat_curr::Float64
    ) where {T<:Union{Int,String}}
    
    β = mcmc.β
    if rand() < β
        was_accepted, suff_stat_curr = double_iex_multinomial_edit_accept_reject!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc, P, 
            aux_data,
            suff_stat_curr
        )
        acc_count[1] += was_accepted
        count[1] += 1
    else 
        was_accepted, suff_stat_curr = double_iex_trans_dim_accept_reject!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc, 
            aux_data,
            suff_stat_curr
        )
        acc_count[2] += was_accepted 
        count[2] += 1
    end 
    return suff_stat_curr
end 

function accept_reject_gamma!(
    γ_curr::Float64,
    S_curr::InteractionSequence{T},
    posterior::SisPosterior{T},
    suff_stat_curr::Float64, 
    mcmc::SisIexInsertDeleteEdit{T},
    aux_data::InteractionSequenceSample{T}
    ) where {T<:Union{Int,String}}

    ε = mcmc.ε
    aux_mcmc = mcmc.aux_mcmc

    γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

    aux_model = SIS(
        S_curr, γ_prop, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, posterior.K_outer
        )
    draw_sample!(aux_data, aux_mcmc, aux_model)

    # Accept reject

    log_lik_ratio = (γ_curr - γ_prop) * suff_stat_curr
    aux_log_lik_ratio = (γ_prop - γ_curr) * sum_of_dists(aux_data, S_curr, posterior.dist)

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
    sample_out_S::Union{InteractionSequenceSample{T},SubArray},
    sample_out_gamma::Union{Vector{Float64},SubArray},
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{T}}=sample_frechet_mean(posterior.data, posterior.dist),
    gamma_init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    if loading_bar
        iter = Progress(
            length(sample_out_S), # How many iters 
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
    γ_curr = copy(gamma_init)

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 1 # Keeps track all samples (included lags and burn_ins) 

    acc_count = [0,0]
    count = [0,0]
    γ_acc_count = 0
    γ_count = 0

    aux_data = [[T[]] for i in 1:posterior.sample_size]
    # Initialise the aux_data 
    aux_model = SIS(
        S_curr, γ_curr, 
        posterior.dist, 
        posterior.V, 
        posterior.K_inner, 
        posterior.K_outer)
    draw_sample!(aux_data, aux_mcmc, aux_model)
    # Initialise sufficient statistic
    suff_stat_curr = mapreduce(
        x -> posterior.dist(S_curr, x), 
        +, 
        posterior.data
        )
    # Get informed proposal matrix
    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)

    while sample_count ≤ length(sample_out_S)
        # Store values
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out_S[sample_count] = deepcopy(S_curr)
            sample_out_gamma[sample_count] = copy(γ_curr)
            sample_count += 1
        end 

        # Update mode
        # ----------- 
        suff_stat_curr = accept_reject_mode!(
            S_curr, S_prop, 
            posterior, γ_curr, 
            mcmc, P, 
            aux_data, 
            acc_count, count,
            suff_stat_curr
        )
        # Update gamma 
        # ------------
        γ_curr, tmp =  accept_reject_gamma!(
            γ_curr,
            S_curr,
            posterior, 
            suff_stat_curr, 
            mcmc, 
            aux_data
        )
        γ_acc_count += tmp
        next!(iter)
        i += 1
    end 

    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 

    ed_acc_prob = acc_count[1]/count[1]
    td_acc_prob = acc_count[2]/count[2]
    γ_acc_prob = γ_acc_count / sum(count)
    return ed_acc_prob, td_acc_prob, γ_acc_prob
end 

function draw_sample(
    mcmc::SisIexInsertDeleteEdit{T},
    posterior::SisPosterior{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{T}=sample_frechet_mean(posterior.data, posterior.dist),
    gamma_init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    sample_out_S = Vector{InteractionSequence{T}}(undef, desired_samples)
    sample_out_gamma = Vector{Float64}(undef, desired_samples)

    draw_sample!(
        sample_out_S,
        sample_out_gamma, 
        mcmc, 
        posterior, 
        burn_in=burn_in, lag=lag, 
        S_init=S_init, gamma_init=gamma_init,
        loading_bar=loading_bar
    )

    return (S=sample_out_S, gamma=sample_out_gamma)
end 

function (mcmc::SisIexInsertDeleteEdit)(
    posterior::SisPosterior{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{T}=sample_frechet_mean(posterior.data, posterior.dist),
    gamma_init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{Int,String}}

    sample_out_S = Vector{InteractionSequence{T}}(undef, desired_samples)
    sample_out_gamma = Vector{Float64}(undef, desired_samples)

    ed_acc_prob, td_acc_prob, γ_acc_prob = draw_sample!(
        sample_out_S,
        sample_out_gamma, 
        mcmc, 
        posterior, 
        burn_in=burn_in, lag=lag, 
        S_init=S_init, gamma_init=gamma_init,
        loading_bar=loading_bar
    )

    p_measures = Dict(
            "Dipsersion acceptance probability" => γ_acc_prob,
            "Edit allocation acceptance probability" => ed_acc_prob,
            "Trans-dimensional acceptance probability" => td_acc_prob
        )

    return SisPosteriorMcmcOutput(
        sample_out_S, 
        sample_out_gamma, 
        posterior,
        p_measures
    )
    
end 