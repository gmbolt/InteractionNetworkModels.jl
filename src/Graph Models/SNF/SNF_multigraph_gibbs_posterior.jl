
function exchange_accept_reject_entry!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    γ_curr::Float64,
    i::Int, 
    mcmc::T,
    posterior::MultigraphSnfPosterior,
    aux_data::Vector{Vector{Int}},
    suff_stat_curr::Float64,
    aux_init_at_prev::Bool
    ) where {T<:Union{SnfExchangeRandGibbs}}

    aux_mcmc = mcmc.aux_mcmc
    data = posterior.data
    dist = posterior.dist

    # Proposal generation 
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:mcmc.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    # Eval acceptance probability

    # Auxiliary sampling 
    G_prop = vec_to_adj_mat(
        x_prop, 
        directed=posterior.directed, self_loops=posterior.self_loops
    )
    aux_model = SNF(G_prop, γ_curr)

    if aux_init_at_prev
        tmp = copy(aux_data[end])
        draw_sample!(aux_data, aux_mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux_data, aux_mcmc, aux_model)
    end 

    aux_log_lik_ratio = γ_curr * (
        sum(x->dist(x,x_curr), aux_data) - sum(x->dist(x,x_prop), aux_data) 
    )

    suff_stat_prop = sum(x->dist(x,x_prop), data)

    log_lik_ratio = γ_curr * (suff_stat_curr - suff_stat_prop)

    log_prior = G_prior.γ * (dist(x_curr, G_prior.mode) - dist(x_prop, G_prior.mode))
    
    log_α = log_lik_ratio + log_prior + aux_log_lik_ratio

    if log(rand()) < log_α
        @inbounds x_curr[i] = w_prop
        return true, suff_stat_prop
    else 
        @inbounds x_prop[i] = w_curr
        return false, suff_stat_curr
    end 
end 

# Optimised version for the hamming distances between mutlisets
function accept_reject_entry_hamming!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    i::Int, 
    mcmc::T,
    model::MultigraphSNF,
    x_mode::Vector{Int} # Must also pass vectorised mode
    ) where {T<:Union{SnfMcmcRandGibbs,SnfMcmcSysGibbs}}

    # Proposal generation 
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:mcmc.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    # Eval acceptance probability
    γ, d = (model.γ, model.d)
    @inbounds w_mode = x_mode[i]
    log_α = γ * (abs(w_curr - w_mode) - abs(w_prop - w_mode))
    
    if log(rand()) < log_α
        @inbounds x_curr[i] = w_prop
        return true 
    else 
        @inbounds x_prop[i] = w_curr
        return false
    end 
end 
