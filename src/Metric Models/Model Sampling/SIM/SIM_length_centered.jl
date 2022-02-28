using Distributions

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::SimMcmcInsertDeleteLengthCentered,
    ind::AbstractVector{Int},
    V::UnitRange,
    K_in_ub::Int
    ) 

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = truncated(Poisson(mean(length, S_curr)), 1, Inf)

    log_ratio = 0.0 
    for i in ind 
        tmp = popfirst!(prop_pointers)
        m = rand(len_dist)
        resize!(tmp, m)
        sample!(V, tmp)
        @inbounds insert!(S_prop, i, tmp)
        log_ratio += - logpdf(len_dist, m) + m*log(length(V)) - Inf * (m > K_in_ub)
    end 
    log_ratio += log(ν_td) - log(min(ν_td,N)) 
    return log_ratio 

end 

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::SimMcmcInsertDeleteLengthCentered,
    ε::Int,
    V::UnitRange,
    K_in_ub::Int
    )

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    len_dist = truncated(Poisson(mean(length, S_curr)), 1, Inf)
    N = length(S_curr)
    log_ratio = 0.0
    ind = mcmc.ind_td

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
    log_ratio += log(ν_td) - log(min(ν_td,N)) 
    return log_ratio 

end 


function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mcmc::SimMcmcInsertDeleteLengthCentered,
    V::UnitRange
    ) 

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = truncated(Poisson(mean(length, S_curr)), 1, Inf)
    log_ratio = 0.0
    ind = mcmc.ind_td

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

    log_ratio += log(min(ν_td,N)) - log(ν_td)
    return log_ratio

end 

function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int}, 
    mcmc::SimMcmcInsertDeleteLengthCentered,
    ind::AbstractVector{Int}, 
    V::UnitRange
    ) 

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = truncated(Poisson(mean(length, S_curr)), 1, Inf)

    log_ratio = 0.0

    for i in Iterators.reverse(ind)
        @inbounds tmp = popat!(S_prop, i)
        pushfirst!(prop_pointers, tmp)
        m = length(tmp)
        log_ratio += logpdf(len_dist, m) - m * log(length(V))
    end 

    log_ratio += log(min(ν_td,N)) - log(ν_td)
    return log_ratio

end 