export SisAuxTermEvaluator, aux_term_current, aux_term_new
export initialise! 

struct SisAuxTermEvaluator{T<:SisMcmcSampler}
    mcmc::T
    d::Metric 
    V::UnitRange{Int}
    K_inner::DimensionRange
    K_outer::DimensionRange
    S_curr::InteractionSequence{Int}
    S_prop::InteractionSequence{Int}
    function SisAuxTermEvaluator(
        mcmc::SisMcmcSampler,
        d::Metric,
        V::UnitRange{Int},
        K_inner::DimensionRange,
        K_outer::DimensionRange
        )
        new{typeof(mcmc)}(
            mcmc, 
            d, V, K_inner, K_outer, 
            InteractionSequence{Int}(), InteractionSequence{Int}()
        )
    end 
end 

function initialise!(
    aux_eval::SisAuxTermEvaluator,
    S::InteractionSequence{Int}
    )
    S_curr = aux_eval.S_curr
    S_prop = aux_eval.S_prop 
    curr_pointers = aux_eval.mcmc.curr_pointers
    prop_pointers = aux_eval.mcmc.prop_pointers

    for i in 1:length(S)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], S[i])
        copy!(S_prop[i], S[i])
    end 
end 
function aux_term_current(
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence, 
    γ_fixed::Float64,
    dist::Metric,
    V::UnitRange,
    K_inner::DimensionRange, K_outer::DimensionRange,
    aux_mcmc::SisMcmcSampler,
    aux_data::InteractionSequenceSample{Int},
    )

    aux_model = SIS(
        S_prop, γ_fixed, 
        dist, 
        V, 
        K_inner, 
        K_outer
    )

    draw_sample!(aux_data, aux_mcmc, aux_model, init=S_prop)

    aux_log_lik_ratio = γ_fixed * (
        sum(dist(x, S_prop) for x in aux_data)
        - sum(dist(x, S_curr) for x in aux_data)
    )

    return aux_log_lik_ratio
end 

function aux_term_new(
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence, 
    γ_fixed::Float64,
    n::Int,
    aux_eval::SisAuxTermEvaluator
    )

    dist, V, K_inner, K_outer = (
        aux_eval.d, 
        aux_eval.V,    
        aux_eval.K_inner, 
        aux_eval.K_outer
    )
    mcmc = aux_eval.mcmc 
    burn_in, lag = (mcmc.burn_in, mcmc.lag)

    S_curr_aux = aux_eval.S_curr
    S_prop_aux = aux_eval.S_prop

    dist_curr = mcmc.dist_curr
    dist_curr[1] = dist(S_curr_aux, S_prop)

    sample_count = 1
    i = 0
    acc_count = 0
    aux_log_lik_ratio = 0.0

    # out = []

    while sample_count ≤ n
        i += 1 
        # Increment the auxiliary term 
        if (i > burn_in) & (((i-1) % lag)==0)
            aux_log_lik_ratio += dist_curr[1] - dist(S_curr_aux, S_curr)
            sample_count += 1
            # push!(out, deepcopy(S_curr_aux))
        end 
        # Accept reject
        acc_count += accept_reject!(
            S_curr_aux, S_prop_aux, 
            S_prop, γ_fixed, dist, V, K_inner, K_outer,
            mcmc
        )
    end 
    aux_log_lik_ratio *= γ_fixed
    # for i in 1:length(S_curr_aux)
    #     migrate!(curr_pointers, S_curr_aux, 1, 1)
    #     migrate!(prop_pointers, S_prop_aux, 1, 1)
    # end 
    return aux_log_lik_ratio
end 

function aux_term_new(
    γ_curr::Float64, 
    γ_prop::Float64, 
    S_fixed::InteractionSequence{Int},
    n::Int,
    aux_eval::SisAuxTermEvaluator
    )

    dist, V, K_inner, K_outer = (
        aux_eval.d, 
        aux_eval.V,    
        aux_eval.K_inner, 
        aux_eval.K_outer
    )
    mcmc = aux_eval.mcmc 
    burn_in, lag = (mcmc.burn_in, mcmc.lag)

    S_curr_aux = aux_eval.S_curr
    S_prop_aux = aux_eval.S_prop

    dist_curr = mcmc.dist_curr
    dist_curr[1] = dist(S_curr_aux, S_fixed)

    sample_count = 1
    i = 0
    aux_log_lik_ratio = 0.0

    # out = []

    while sample_count ≤ n
        i += 1 
        # Increment the auxiliary term 
        if (i > burn_in) & (((i-1) % lag)==0)
            aux_log_lik_ratio += dist_curr[1]
            sample_count += 1
        end 
        # Accept reject
        accept_reject!(
            S_curr_aux, S_prop_aux, 
            S_fixed, γ_prop, dist, V, K_inner, K_outer,
            mcmc
        )
    end 
    aux_log_lik_ratio *= (γ_prop - γ_curr) 
    return aux_log_lik_ratio
end 