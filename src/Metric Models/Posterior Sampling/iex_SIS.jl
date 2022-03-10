using Distributions, StatsBase, ProgressMeter, InvertedIndices, IterTools

export draw_sample_mode!, draw_sample_mode
export draw_sample_gamma!, draw_sample_gamma
export rand_multivariate_bernoulli, rand_multinomial_dict
export flip!, multinomial_flip!, flip_informed!
export rand_restr_bins, flip_informed_excl!


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

function rand_multinomial_dict(μ_cusum::Vector{Float64}, ntrials::Int)
    @assert μ_cusum[1] ≈ 0 "First entry must be 0.0 (for differencing to find probabilities)."
    out = Dict{Int,Int}()
    for i in 1:ntrials
        β = rand()
        j = findfirst(x->x>β, μ_cusum)
        out[j-1] = get(out, j-1, 0) + 1
    end 
    return out
end 

function rand_restr_bins(
    bins::Vector{Int},
    n::Int
    )
    cum_bins = cumsum(bins)
    @assert n <= cum_bins[end] "Cannot sample more than capacity of bins."
    ind_flat = zeros(Int, n)
    StatsBase.seqsample_a!(1:cum_bins[end], ind_flat)
    out = Dict{Int,Int}()
    for i in ind_flat 
        j = findfirst(x->x>=i, cum_bins)
        out[j] = get(out,j,0) + 1 
    end 
    return out
end 

function rand_restr_cum_bins(
    cum_bins::Vector{Int},
    n::Int
    )
    @assert n <= cum_bins[end] "Cannot sample more than capacity of bins."
    ind_flat = zeros(Int, n)
    StatsBase.seqsample_a!(1:cum_bins[end], ind_flat)
    out = Dict{Int,Int}()
    for i in ind_flat 
        j = findfirst(x->x>=i, cum_bins)
        out[j] = get(out,j,0) + 1 
    end 
    return out
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


function flip!(
    x::Path, 
    ind_flip::AbstractArray{Int},
    V::Vector{Int}
    )   
    
    # Get unique vertices not being flipped 
    for i in ind_flip
        x[i] = curr_val 
        curr_vert_ind = findfirst(x->x==curr_val, V)
        tmp = rand(1:(length(V)-1))

        if tmp >= curr_vert_ind
            x[i] = V[tmp+1]
        else 
            x[i] = V[tmp]
        end 

    end 

end 

function flip!(
    x::Path, 
    ind_flip::AbstractArray{Int},
    V::UnitRange{Int}
    )   
    
    # Get unique vertices not being flipped 
    for i in ind_flip
        x[i] = curr_val 
        tmp = rand(1:(length(V)-1))
        if tmp >= curr_val
            x[i] = tmp+1
        else 
            x[i] = tmp
        end 
    end 

end 

function flip_informed!(
    x::Path, 
    ind_flip::AbstractArray{Int},
    P::CumCondProbMatrix
    )

    curr_vertices = unique(view(x, Not(ind_flip)))
    if length(curr_vertices) == 0 
        V = size(P)[2]
        tmp = fill(1/V, V)
        pushfirst!(tmp, 0.0)
        μ_cusum = cumsum(tmp)
    else 
        μ_cusum = sum(P[:,curr_vertices], dims=2)[:] ./ length(curr_vertices)
    end 
    log_ratio = 0.0
    for index in ind_flip 
        curr_val = x[index]
        x[index], prob = rand_multivariate_bernoulli(μ_cusum)
        log_ratio += (
            log(μ_cusum[curr_val+1]-μ_cusum[curr_val]) - log(prob) 
        )
    end 
    return log_ratio
end 


function flip_informed!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    mcmc::Union{SisIexInsertDelete,SimIexInsertDelete},
    P::CumCondProbMatrix
    ) 

    δ = rand(1:mcmc.ν_ed)
    alloc = rand_restr_bins(length.(S_curr), δ)
    ind = mcmc.ind_add
    log_ratio = 0.0

    for (key,val) in pairs(alloc)
        ind_flip = view(ind, 1:val)
        log_ratio += flip_informed!(
            S_prop[key], 
            ind_flip, 
            P
        ) 
    end 
    return log_ratio

end

function flip_informed_excl!(
    x::Path, 
    ind_flip::AbstractArray{Int},
    P::CumCondProbMatrix
    )

    curr_vertices = unique(view(x, Not(ind_flip)))
    if length(curr_vertices) == 0 
        V = size(P)[2]
        tmp = fill(1/V, V)
        pushfirst!(tmp, 0.0)
        μ_cusum = cumsum(tmp)
    else 
        μ_cusum = sum(P[:,curr_vertices], dims=2)[:] ./ length(curr_vertices)
    end 
    log_ratio = 0.0
    for index in ind_flip 
        curr_val = x[index]
        prop_val = curr_val
        prop_prob = 0.0
        while prop_val == curr_val
            prop_val, prop_prob = rand_multivariate_bernoulli(μ_cusum)
        end 
        x[index] = prop_val 
        curr_prob = μ_cusum[curr_val+1]-μ_cusum[curr_val]
        log_ratio += (
            log(curr_prob) + log(1-curr_prob)
            - log(prop_prob) - log(1-prop_prob)
        )
    end 
    return log_ratio
end 

function flip_informed_excl!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    mcmc::Union{SisIexInsertDelete,SimIexInsertDelete},
    P::CumCondProbMatrix
    ) 

    δ = rand(1:mcmc.ν_ed)
    alloc = rand_restr_bins(length.(S_curr), δ)
    ind = mcmc.ind_add
    log_ratio = 0.0

    for (key,val) in pairs(alloc)
        ind_flip = view(ind, 1:val)
        log_ratio += flip_informed_excl!(
            S_prop[key], 
            ind_flip, 
            P
        ) 
    end 
    return log_ratio

end

function double_iex_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SisPosterior,
    γ_curr::Float64,
    mcmc::SisIexInsertDelete,
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
            # a,b = (lb(n, δ_tmp, model), ub(n, δ_tmp))
            d = rand(0:min(n-K_in_lb, δ_tmp))
            m = n + δ_tmp - 2*d

            # Catch invalid proposals
            if  (m > K_in_ub) 
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
            log_ratio += log(min(n-K_in_lb, δ_tmp)+1) - log(min(m-K_in_lb, δ_tmp)+1)

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

    if aux_init_at_prev
        @inbounds tmp = deepcopy(aux_data[end])
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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

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


function double_iex_flip_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SisPosterior,
    γ_curr::Float64,
    mcmc::SisIexInsertDelete,
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
        StatsBase.seqsample_a!(1:lengths[key], ind_flip)
        log_ratio += flip_informed_excl!(
            S_prop[key], 
            ind_flip, 
            P
        ) 
    end 
    
    aux_model = SIS(
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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in keys(alloc)
            copy!(S_curr[i], S_prop[i])
        end
        return 1, suff_stat_prop
    else 
        for i in keys(alloc)
            copy!(S_prop[i], S_curr[i])
        end 
        return 0, suff_stat_curr
    end 

end 

# Insertion/deletion proposal generation functions 
# ================================================

# With vector of indices (uninformed)
# -----------------------------------
function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T,
    ind::AbstractVector{Int},
    V::UnitRange, 
    K_in_ub::Int
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = mcmc.len_dist

    log_ratio = 0.0 
    for i in ind 
        tmp = popfirst!(prop_pointers)
        m = rand(len_dist)
        resize!(tmp, m)
        sample!(V, tmp)
        insert!(S_prop, i, tmp)
        log_ratio += - logpdf(len_dist, m) + m*log(length(V)) - Inf * (m > K_in_ub)
    end 
    return log_ratio 

end 

function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mcmc::T,
    ind::AbstractVector{Int},
    V::UnitRange
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    len_dist = mcmc.len_dist
    N = length(S_curr)

    log_ratio = 0.0

    for i in Iterators.reverse(ind)
        tmp = popat!(S_prop, i)
        pushfirst!(prop_pointers, tmp)
        m = length(tmp)
        log_ratio += logpdf(len_dist, m) - m * log(length(V))
    end 
    return log_ratio

end 

# With number of insertions/deletions
# -----------------------------------
function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T,
    ε::Int,
    V::UnitRange,
    K_in_ub::Int
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    len_dist = mcmc.len_dist
    log_ratio = 0.0
    ind = mcmc.ind_td_add
    
    n = length(S_prop)+ε
    k = ε
    i = 0 
    j = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i += 1
        j += 1
        # Insert path at index i
        # i is now index to insert
        ind[j] = i
        tmp = popfirst!(prop_pointers) # Get storage 
        m = rand(len_dist)  # Sample length 
        resize!(tmp, m) # Resize 
        sample!(V, tmp) # Sample new entries uniformly
        @inbounds insert!(S_prop, i, tmp) # Insert path into S_prop
        log_ratio += - logpdf(len_dist, m) + m*log(length(V)) - Inf * (m > K_in_ub)  # Add to log_ratio term
        n -= 1
        k -= 1
    end
    return log_ratio 

end 

function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mcmc::T,
    ε::Int,
    V::UnitRange
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    len_dist = mcmc.len_dist
    log_ratio = 0.0
    ind = mcmc.ind_td_del # Store which entries were deleted 

    n = length(S_curr)
    k = ε   
    i = 0 
    j = 0
    live_index = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        j+=1
        # Delete path 
        # i is now index to delete 
        @inbounds ind[j] = i
        @inbounds tmp = popat!(S_prop, i - live_index)
        pushfirst!(prop_pointers, tmp)
        m = length(tmp)
        log_ratio += logpdf(len_dist, m) - m * log(length(V))
        live_index += 1
        n -= 1
        k -= 1
    end
    return log_ratio
end 

# Informed insertions/deletions 
# -----------------------------

function imcmc_multi_insert_prop_sample_informed!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T,
    ε::Int,
    p_ins::Categorical, 
    K_in_ub::Int
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    len_dist = mcmc.len_dist
    log_ratio = 0.0
    ind = mcmc.ind_td_add

    n = length(S_curr)+ε
    k = ε
    i = 0 
    j = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i += 1
        j += 1
        # Insert path at index i
        # i is now index to insert
        ind[j] = i
        tmp = popfirst!(prop_pointers) # Get storage 
        m = rand(len_dist)  # Sample length 
        resize!(tmp, m) # Resize 
        for i in eachindex(tmp) # Sample new entries
            v = rand(p_ins)
            tmp[i] = v 
            log_ratio += - logpdf(p_ins, v)
        end 
        @inbounds insert!(S_prop, i, tmp) # Insert path into S_prop
        log_ratio += - logpdf(len_dist, m) - Inf * (m > K_in_ub)  # Add to log_ratio term
        n -= 1
        k -= 1
    end
    return log_ratio 

end 

function imcmc_multi_delete_prop_sample_informed!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mcmc::T,
    ε::Int,
    p_ins::Categorical
    ) where {T<:Union{SisPosteriorSampler,SimPosteriorSampler}}

    prop_pointers = mcmc.prop_pointers
    len_dist = mcmc.len_dist
    log_ratio = 0.0
    ind = mcmc.ind_td_del # Store which entries were deleted 

    n = length(S_curr)
    k = ε   
    i = 0 
    j = 0
    live_index = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        j+=1
        # Delete path 
        # i is now index to delete 
        @inbounds ind[j] = i
        @inbounds tmp = popat!(S_prop, i - live_index)
        pushfirst!(prop_pointers, tmp)
        m = length(tmp)
        for v in tmp
            log_ratio += logpdf(p_ins,v)
        end 
        log_ratio += logpdf(len_dist, m) 
        # Update counters 
        live_index += 1
        n -= 1
        k -= 1
    end
    return log_ratio
end 


function double_iex_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    posterior::SisPosterior, 
    γ_curr::Float64,
    mcmc::SisIexInsertDelete,
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
    aux_model = SIS(
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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

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
    posterior::SisPosterior, 
    γ_curr::Float64,
    mcmc::SisIexInsertDelete,
    p_ins::Categorical,
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
    aux_model = SIS(
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

    # Log acceptance probability
    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    # @show is_insert, suff_stat_curr, suff_stat_prop, exp(log_α)
    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
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

# =========================
# Samplers 
# =========================

# Mode conditional 
# ----------------

function draw_sample_mode!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior,
    γ_fixed::Float64;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    ) 

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
    suff_stats = Float64[suff_stat_curr] # Storage for all sufficient stats (for diagnostics)

    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)
    p_ins = get_informed_insertion_dist(posterior, mcmc.α)
    while sample_count ≤ length(sample_out)
        i += 1
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = deepcopy(S_curr)
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
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior,
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true, 
    aux_init_at_prev::Bool=false
    ) 

    sample_out = Vector{Vector{Path{Int}}}(undef, desired_samples)
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

function (mcmc::SisIexInsertDelete)(
    posterior::SisPosterior, 
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist),
    loading_bar::Bool=true, 
    aux_init_at_prev::Bool=false
    ) 
    sample_out = Vector{Vector{Path{Int}}}(undef, desired_samples)

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
    output = SisPosteriorModeConditionalMcmcOutput(
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
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior,
    S_fixed::InteractionSequence{Int};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64=4.0,
    loading_bar::Bool=true, 
    aux_init_at_prev::Bool=false
    )

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
    γ_curr = γ_init
    aux_data = [[Int[]] for i in 1:posterior.sample_size]

    # Evaluate sufficient statistic
    suff_stat = sum(
        x -> posterior.dist(S_curr, x), 
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
        if aux_init_at_prev
            tmp = deepcopy(aux_data[end])
            draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
        else 
            draw_sample!(aux_data, aux_mcmc, aux_model)
        end 

        # Accept reject

        log_lik_ratio = (γ_curr - γ_prop) * suff_stat

        dist = posterior.dist
        aux_log_lik_ratio = (γ_prop - γ_curr) * sum(x->dist(x,S_curr),aux_data)

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
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior,
    S_fixed::InteractionSequence{Int};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64,
    loading_bar::Bool=true, 
    aux_init_at_prev::Bool=false
    ) 

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


function (mcmc::SisIexInsertDelete)(
    posterior::SisPosterior, 
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

    output = SisPosteriorDispersionConditionalMcmcOutput(
            S_fixed, 
            sample_out, 
            posterior,
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
    posterior::SisPosterior,
    γ_curr::Float64, 
    mcmc::SisIexInsertDelete,
    P::CumCondProbMatrix,
    p_ins::Categorical,
    aux_data::InteractionSequenceSample{Int},
    acc_count::Vector{Int},
    count::Vector{Int},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
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
            posterior, 
            γ_curr, 
            mcmc, 
            p_ins,
            aux_data,
            suff_stat_curr,
            aux_init_at_prev
        )
        
        acc_count[3] += was_accepted 
        count[3] += 1
    end 
    return suff_stat_curr
end 

function accept_reject_gamma!(
    γ_curr::Float64,
    S_curr::InteractionSequence{Int},
    posterior::SisPosterior,
    mcmc::SisIexInsertDelete,
    aux_data::InteractionSequenceSample{Int},
    suff_stat_curr::Float64, 
    aux_init_at_prev::Bool
    ) 

    ε = mcmc.ε
    aux_mcmc = mcmc.aux_mcmc

    γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

    aux_model = SIS(
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

    log_lik_ratio = (γ_curr - γ_prop) * suff_stat_curr
    dist = posterior.dist
    aux_log_lik_ratio = (γ_prop - γ_curr) * sum(x->dist(x,S_curr),aux_data)

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
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist),
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    )

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
    γ_curr = copy(γ_init)

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 1 # Keeps track all samples (included lags and burn_ins) 

    acc_count = [0,0,0]
    count = [0,0,0]
    γ_acc_count = 0

    aux_data = [[Int[]] for i in 1:posterior.sample_size]
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
    suff_stats = Float64[suff_stat_curr]
    # Get informed proposal matrix
    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, mcmc.α)
    p_ins = get_informed_insertion_dist(posterior, mcmc.α)
    while sample_count ≤ length(sample_out_S)
        # Store values
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out_S[sample_count] = deepcopy(S_curr)
            sample_out_gamma[sample_count] = copy(γ_curr)
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

    ed_acc_prob = acc_count[1]/count[1]
    flip_acc_prob = acc_count[2]/count[2]
    td_acc_prob = acc_count[3]/count[3]
    γ_acc_prob = γ_acc_count / sum(count)
    return ed_acc_prob, flip_acc_prob, td_acc_prob, γ_acc_prob, suff_stats
end 

function draw_sample(
    mcmc::SisIexInsertDelete,
    posterior::SisPosterior;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::InteractionSequence{Int}=sample_frechet_mean(posterior.data, posterior.dist),
    γ_init::Float64=5.0,
    loading_bar::Bool=true,
    aux_init_at_prev::Bool=false
    )

    sample_out_S = Vector{InteractionSequence}(undef, desired_samples)
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

function (mcmc::SisIexInsertDelete)(
    posterior::SisPosterior;
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

    return SisPosteriorMcmcOutput(
        sample_out_S, 
        sample_out_gamma, 
        posterior,
        suff_stats,
        p_measures
    )
    
end 