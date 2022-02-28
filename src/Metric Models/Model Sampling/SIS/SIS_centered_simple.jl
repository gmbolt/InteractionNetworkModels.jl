using Distributions, StatsBase

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    mcmc::SisMcmcInsertDeleteCenter,
    ind::AbstractVector{Int},
    V::UnitRange,
    K_in_ub::Int
    ) 

    prop_pointers = mcmc.prop_pointers
    ν_td = mcmc.ν_td
    N = length(S_curr)
    len_dist = mcmc.len_dist
    
    # Make insertion Categorical distribution ...
    # p_ins = 
    mean_len = floor(Int,mean(length.(S)))
    log_ratio = 0.0 
    for i in ind 
        tmp = popfirst!(prop_pointers)
        m = rand(len_dist)
        resize!(tmp, m)
        sample!(p_ins, tmp)
        insert!(S_prop, i, tmp)
        log_ratio += - logpdf(len_dist, m) + m*log(length(V)) - Inf * (m > K_in_ub)
    end 
    log_ratio += log(ν_td) - log(min(ν_td,N)) 
    return log_ratio 

end 