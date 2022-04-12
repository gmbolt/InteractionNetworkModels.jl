using Distributions, StatsBase

export imcmc_multinomial_edit_accept_reject!, unif_multinomial_sample_tester
export imcmc_trans_dim_accept_reject!, draw_sample!, draw_sample
export rand_delete!, rand_insert!, accept_reject!

function rand_delete!(x::Path{Int}, d::Int)

    n = length(x)
    k = d
    i = 0 
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
        # i is now index to delete 
        deleteat!(x, i-live_index)
        live_index += 1
        n -= 1
        k -= 1
    end

end 

function rand_insert!(x::Path{Int}, a::Int, V::UnitRange)

    n = length(x)+a
    k = a
    i = 0 
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        # i is now index to insert
        insert!(x, i, rand(V))
        n -= 1
        k -= 1
    end
end 

# Edit Allocation Move 
# --------------------

function imcmc_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mode::InteractionSequence{Int}, 
    γ::Float64, 
    dist::Metric, 
    V::UnitRange, 
    K_inner::DimensionRange, 
    K_outer::DimensionRange,
    mcmc::Union{SisMcmcInsertDelete,SisMcmcSplitMerge}
    ) 

    dist_curr = mcmc.dist_curr

    N = length(S_curr)  
    K_in_lb = K_inner.l
    K_in_ub = K_inner.u
    δ = rand(1:mcmc.ν_ed)  # Number of edits to enact 
    rem_edits = δ # Remaining edits to allocate
    len_diffs = 0
    j = 0 # Keeps track how many interaction have been edited 
    log_prod_term = 0.0 

    # println("Making $δ edits in total...")

    for i in eachindex(S_curr)

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
                    copy!(S_prop[i], S_curr[i])
                end 
                return 0 
            end 

            I_tmp = S_prop[i]
            rand_delete!(I_tmp, d)
            rand_insert!(I_tmp, δ_tmp-d, V)

            mcmc.ind_update[j] = i # Store which interaction was updated
            
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
    log_ratio = log(length(V)) * len_diffs + log_prod_term
    # log_ratio = log_dim_diff + log_prod_term
    dist_prop = dist(mode, S_prop)
    log_lik_ratio = -γ * (
        dist_prop - dist_curr[1]
    )

    # Log acceptance probability
    log_α = log_lik_ratio + log_ratio

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_curr[i], S_prop[i])
        end
        dist_curr[1] = dist_prop # Update stored distance
        return 1 
    else 
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_prop[i], S_curr[i])
        end 
        return 0 
    end 
end 

function imcmc_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    model::SIS, 
    mcmc::Union{SisMcmcInsertDelete,SisMcmcSplitMerge}
    ) 

    return imcmc_multinomial_edit_accept_reject!(
        S_curr, S_prop, 
        model.mode, model.γ,
        model.dist, model.V, 
        model.K_inner, model.K_outer, 
        mcmc
    )
end 

# To check the above code is doing as required. Output should be samples from a uniform
# with n trials and k bins. 
function unif_multinomial_sample_tester(k, n)
    n_rem = n
    x = Int[]
    for i in 1:(k-1) 
        p = 1/(k-i+1)
        z = rand(Binomial(n_rem, p)) 
        push!(x, z)
        n_rem -= z
    end 
    push!(x, n_rem)
    return x 
end 

# Trans-dimensional Move 
# ----------------------

function migrate!(
    y::Vector{Vector{Int}}, x::Vector{Vector{Int}},
    j::Int, i::Int)
    @inbounds insert!(y, j, x[i])
    @inbounds deleteat!(x, i)
end 

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T,
    ind::AbstractVector{Int},
    V::UnitRange,
    K_in_ub::Int
    ) where {T<:Union{SisMcmcSampler,SimMcmcSampler}}

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
        @inbounds insert!(S_prop, i, tmp)
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
    ) where {T<:Union{SisMcmcSampler,SimMcmcSampler}}

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = mcmc.len_dist

    log_ratio = 0.0

    for i in Iterators.reverse(ind)
        @inbounds tmp = popat!(S_prop, i)
        pushfirst!(prop_pointers, tmp)
        m = length(tmp)
        log_ratio += logpdf(len_dist, m) - m * log(length(V))
    end 
    return log_ratio

end 


function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::T,
    ε::Int,
    V::UnitRange,
    K_in_ub::Int
    ) where {T<:Union{SisMcmcSampler,SimMcmcSampler}}

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
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
    ) where {T<:Union{SisMcmcSampler,SimMcmcSampler}}

    prop_pointers = mcmc.prop_pointers
    N = length(S_curr)
    len_dist = mcmc.len_dist
    log_ratio = 0.0
    ind = mcmc.ind_td_del # Store which entries were deleted 

    n = length(S_prop)
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

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    mode::InteractionSequence{Int}, 
    γ::Float64, 
    dist::Metric, 
    V::UnitRange, 
    K_inner::DimensionRange, 
    K_outer::DimensionRange,
    mcmc::Union{SisMcmcInsertDeleteGibbs,SisMcmcInsertDelete}
    ) 

    K_out_lb = K_outer.l
    K_out_ub = K_outer.u
    K_in_ub = K_inner.u
    ν_td = mcmc.ν_td
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers

    dist_curr = mcmc.dist_curr

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    ε = rand(1:ν_td)
    d = rand(0:min(ε, N))
    a = ε - d

    # Catch invalid proposal (outside dimension bounds)
    M = N - d + a  # Number of paths in proposal
    if (M < K_out_lb) | (M > K_out_ub)
        return false
    end 
    if d > 0 
        log_ratio +=  imcmc_multi_delete_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            d, 
            V
        ) # Enact deletion move and catch log ratio term 
    end
    if a > 0  # Number of insertions 
        log_ratio += imcmc_multi_insert_prop_sample!(
            S_curr, S_prop, 
            mcmc, 
            a, 
            V, K_in_ub
        )
    end 

    log_ratio += log(min(ε, N) + 1) - log(min(ε, M) + 1)

    # Now do accept-reject step 
    dist_prop = dist(mode, S_prop)
    log_α = - γ * (
        dist_prop - dist_curr[1]
    ) + log_ratio

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
        dist_curr[1] = dist_prop  # Update stored distance
        return true 
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
        return false 
    end 
end 

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    model::SIS, 
    mcmc::Union{SisMcmcInsertDeleteGibbs,SisMcmcInsertDelete}
    ) 
    return imcmc_trans_dim_accept_reject!(
        S_curr, S_prop,
        model.mode, model.γ, 
        model.dist, model.V, 
        model.K_inner, model.K_outer, 
        mcmc
    )
end 

function accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    mode::InteractionSequence{Int}, 
    γ::Float64, 
    dist::Metric, 
    V::UnitRange, 
    K_inner::DimensionRange, 
    K_outer::DimensionRange,
    mcmc::Union{SisMcmcInsertDeleteGibbs,SisMcmcInsertDelete}
    )

    β = mcmc.β
    if rand() < β
        return imcmc_multinomial_edit_accept_reject!(
            S_curr, S_prop, 
            mode, γ, dist, V, K_inner, K_outer,
            mcmc
        )
    else 
        return imcmc_trans_dim_accept_reject!(
            S_curr, S_prop, 
            mode, γ, dist, V, K_inner, K_outer,
            mcmc
        )
    end 

end 

function accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    model::SIS, 
    mcmc::SisMcmcInsertDelete
    ) 
    β = mcmc.β
    if rand() < β
        return imcmc_multinomial_edit_accept_reject!(
            S_curr, S_prop, 
            model, mcmc
        )
    else 
        return imcmc_trans_dim_accept_reject!(
            S_curr, S_prop, 
            model, mcmc
        )
    end 

end 

function accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    model::SIS, 
    mcmc::SisMcmcInsertDelete,
    count::Vector{Int},
    acc_count::Vector{Int}
    ) 
    β = mcmc.β
    if rand() < β

        was_acc = imcmc_multinomial_edit_accept_reject!(
            S_curr, S_prop, 
            model, mcmc
        )
        count[1] += 1
        acc_count[1] += was_acc
    else 
        was_acc = imcmc_trans_dim_accept_reject!(
            S_curr, S_prop, 
            model, mcmc
        )
        count[2] += 1
        acc_count[2] += was_acc
    end 

end 

# Sampler Functions 
# -----------------
"""
    draw_sample!(
        sample_out::InteractionSequenceSample, 
        mcmc::SisMcmcInsertDelete, 
        model::SIS;
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::InteractionSequence=get_init(model, mcmc.init)
    )

Draw sample in-place from given SIS model `model::SIS` via MCMC algorithm with edit allocation and interaction insertion/deletion, storing output in `sample_out::InteractionSequenceSample`. 

Accepts keyword arguments to change MCMC output, including burn-in, lag and initial values. If not given, these are set to the default values of the passed MCMC sampler `mcmc::SisMcmcInsertDelete`.
"""
function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SisMcmcInsertDelete,
    model::SIS;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers

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
    count = [0,0]
    acc_count = [0,0]

    mcmc.dist_curr[1] = model.dist(S_curr, model.mode) # Initialise stored distance

    while sample_count ≤ length(sample_out)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end 
        # Accept reject
        accept_reject!(S_curr, S_prop, model, mcmc, count, acc_count)
    end 
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return count, acc_count
end 

"""
    draw_sample(
        mcmc::SisMcmcInsertDelete, 
        model::SIS;
        desired_samples::Int=mcmc.desired_samples, 
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::Vector{Path{T}}=get_init(model, mcmc.init)
        )
"""
function draw_sample(
    mcmc::SisMcmcInsertDelete, 
    model::SIS;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 

function (mcmc::SisMcmcInsertDelete)(
    model::SIS;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    (count, acc_count) = draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)

    p_measures = Dict(
            "Proportion Update Moves" => count[1]/sum(count),
            "Update Move Acceptance Probability" => acc_count[1] / count[1],
            "Trans-Dimensional Move Acceptance Probability" => acc_count[2] / count[2]
        )
    output = SisMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 