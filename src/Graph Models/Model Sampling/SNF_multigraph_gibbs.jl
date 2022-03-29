
function accept_reject_entry!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    i::Int, 
    mcmc::T,
    model::MultigraphSNF,
    x_mode::Vector{Int} # Must also pass vectorised mode
    ) where {T<:Union{SnfMcmcSysGibbs,SnfMcmcRandGibbs}}

    # Proposal generation 
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:mcmc.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    # Eval acceptance probability
    γ, d = (model.γ, model.d)
    log_α = γ * (d(x_curr, x_mode) - d(x_prop, x_mode))

    if log(rand()) < log_α
        @inbounds x_curr[i] = w_prop
        return true 
    else 
        @inbounds x_prop[i] = w_curr
        return false
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



# Systematic Gibbs scan optimised for Hamming distance
function draw_sample_hamming!(
    out::Vector{Vector{Int}},
    mcmc::SnfMcmcSysGibbs, 
    model::MultigraphSNF;
    burn_in::Int=mcmc.burn_in, lag::Int=mcmc.lag,
    init::Vector{Int}=adj_mat_to_vec(model.mode, directed=model.directed, self_loops=model.self_loops)
    )

    x_curr = copy(init)
    x_prop = copy(x_curr)
    x_mode = adj_mat_to_vec(
        model.mode, 
        directed=model.directed, 
        self_loops=model.self_loops
    )
    M = length(x_mode) # Number of possible unqiue edges 
    sample_count = 1
    accept_count = 0
    iter_count = 0
    while sample_count ≤ length(out)
        # Store value if beyond burn-in and coherent with lags
        if (iter_count > burn_in) & (((iter_count-1) % lag)==0)
            @inbounds copy!(out[sample_count], x_curr)
            sample_count += 1
        end 
        # Gibbs scan 
        for i in 1:M
            was_accepted = accept_reject_entry_hamming!(
                x_curr, x_prop,
                i,
                mcmc, model,
                x_mode
            )
            accept_count += was_accepted
            iter_count += 1
        end 

    end 
    return accept_count / iter_count
end 


function draw_sample!(
    out::Vector{Vector{Int}},
    mcmc::SnfMcmcSysGibbs, 
    model::MultigraphSNF;
    burn_in::Int=mcmc.burn_in, lag::Int=mcmc.lag,
    init::Vector{Int}=adj_mat_to_vec(model.mode, directed=model.directed, self_loops=model.self_loops)
    )
    # For Cityblock (what I see as hamming for multisets) we can optimise
    x_curr = copy(init)
    x_prop = copy(x_curr)
    x_mode = adj_mat_to_vec(
        model.mode, 
        directed=model.directed, 
        self_loops=model.self_loops
    )
    M = length(x_mode) # Number of possible unqiue edges 
    sample_count = 1
    accept_count = 0
    iter_count = 0
    while sample_count ≤ length(out)
        # Store value if beyond burn-in and coherent with lags
        if (iter_count > burn_in) & (((iter_count-1) % lag)==0)
            @inbounds copy!(out[sample_count], x_curr)
            sample_count += 1
        end 
        # Gibbs scan 
        for i in 1:M
            was_accepted = accept_reject_entry!(
                x_curr, x_prop,
                i,
                mcmc, model,
                x_mode
            )
            accept_count += was_accepted
            iter_count += 1
        end 

    end 
    return accept_count / iter_count
end 


# Random scann Gibbs 
# -------------------

function draw_sample_hamming!(
    out::Vector{Vector{Int}},
    mcmc::SnfMcmcRandGibbs, 
    model::MultigraphSNF;
    burn_in::Int=mcmc.burn_in, lag::Int=mcmc.lag,
    init::Vector{Int}=adj_mat_to_vec(model.mode, directed=model.directed, self_loops=model.self_loops)
    )

    x_curr = copy(init)
    x_prop = copy(x_curr)
    x_mode = adj_mat_to_vec(
        model.mode, 
        directed=model.directed, 
        self_loops=model.self_loops
    )
    M = length(x_mode) # Number of possible unqiue edges 
    sample_count = 1
    accept_count = 0
    iter_count = 0

    while sample_count ≤ length(out)
        # Store value if beyond burn-in and coherent with lags
        if (iter_count > burn_in) & (((iter_count-1) % lag)==0)
            @inbounds copy!(out[sample_count], x_curr)
            sample_count += 1
        end 
        # Gibbs updates random component
        i = rand(1:M)
        was_accepted = accept_reject_entry_hamming!(
            x_curr, x_prop,
            i,
            mcmc, model,
            x_mode
        )
        accept_count += was_accepted
        iter_count += 1

    end 
    return accept_count / iter_count
end 

function draw_sample!(
    out::Vector{Vector{Int}}, # Vectorised implementation 
    mcmc::SnfMcmcRandGibbs, 
    model::MultigraphSNF;
    burn_in::Int=mcmc.burn_in, lag::Int=mcmc.lag,
    init::Vector{Int}=adj_mat_to_vec(model.mode, directed=model.directed, self_loops=model.self_loops)
    )

    x_curr = copy(init)
    x_prop = copy(x_curr)
    x_mode = adj_mat_to_vec(
        model.mode, 
        directed=model.directed, 
        self_loops=model.self_loops
    )
    M = length(x_mode) # Number of possible unqiue edges 
    sample_count = 1
    accept_count = 0
    iter_count = 0

    while sample_count ≤ length(out)
        # Store value if beyond burn-in and coherent with lags
        if (iter_count > burn_in) & (((iter_count-1) % lag)==0)
            @inbounds copy!(out[sample_count], x_curr)
            sample_count += 1
        end 
        # Gibbs updates random component
        i = rand(1:M)
        was_accepted = accept_reject_entry!(
            x_curr, x_prop,
            i,
            mcmc, model,
            x_mode
        )
        accept_count += was_accepted
        iter_count += 1

    end 
    return accept_count / iter_count
end 

