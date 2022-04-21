using Distances
export delete_insert!, vectorisable_metrics

function get_vectorisable_metrics(model::MultigraphSNF)
    return [Cityblock]
end 

function is_vectorisable(model::MultigraphSNF)
    return typeof(model.d) ∈ get_vectorisable_metrics(model)
end 

# Accept reject for vectorised scheme 
function accept_reject!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    mcmc::SnfMcmcInsertDelete,
    model::MultigraphSNF,
    x_mode::Vector{Int} # Must also pass vectorised mode
    )
    M_curr = sum(x_curr)    # Total number of edges in current state 
    M_prop = M_curr         # Total number of edges in proposal (to vary)        
    δ = rand(1:mcmc.ν)
    d = rand(0:min(δ,M_curr)) # Number of edges to delete
    log_ratio = 0.0 
    
    # Delete edges 
    # This uses random sampling from urn. That is, imagine an urn containing all edges, with multiplicity.
    # We sample edges to delete by drawing from this urn. 
    # See sample_urn_without_replacement() in InteractionNetworkModels\src\Graph Models\utils.jl
    for i in 1:d 
        counter = 0
        z = rand(1:M_prop)
        for (edge, edge_weight) in enumerate(x_prop)
            counter += edge_weight
            if counter ≥ z 
                # We are going to take away a single edge 
                log_ratio -= log(x_prop[edge])
                x_prop[edge] -= 1
                M_prop -= 1
                break
            end 
        end 
    end 
    # Add edges 
    # println("Additons:", δ-d)
    N = length(x_curr)
    for i in 1:(δ-d)
        edge = rand(1:N)
        x_prop[edge] += 1
        log_ratio += log(x_prop[edge])
        M_prop += 1
    end
    # Add log(M_curr!/M_prop!)
    log_size_factorial = 0.0
    if M_curr > M_prop
        for i in (M_prop+1):M_curr 
            log_size_factorial += log(i)
        end 
    elseif M_prop > M_curr
        for i in (M_curr+1):M_prop
            log_size_factorial -= log(i)
        end 
    end
    # Now add term for insertions 
    log_insertion = (M_prop-M_curr)*log(N)     # log((1/V)^(d-a)) = (M_prop-M_curr)*log(V) (n.b. M_curr = M_prop - d + a)
    
    log_ratio += log_size_factorial + log_insertion

    log_ratio += log(min(M_curr, δ)+1) - log(min(M_prop, δ)+1)
    # With log_α denoting the log acceptance probability and
    #       f(x) = exp(- γ d(x, xᵐ))
    # denoting the un-normalised probability of x, we have 
    # log_α = f(x_prop) - f(x_curr) + log_ratio 
    # thus the following...
    γ, d = (model.γ, model.d)
    # println(x_curr==x_prop)
    log_model = γ * (d(x_curr, x_mode) - d(x_prop, x_mode))
    log_α =  log_model + log_ratio
    # @show log_model, log_ratio, M_curr, M_prop, log_size_factorial, log_insertion
    if log(rand()) < log_α 
        copy!(x_curr, x_prop)
        return true 
    else 
        copy!(x_prop, x_curr)
        return false
    end 
end 

function accept_reject_test!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    mcmc::SnfMcmcInsertDelete,
    model::MultigraphSNF,
    x_mode::Vector{Int} # Must also pass vectorised mode
    )
    M_curr = sum(x_curr)    # Total number of edges in current state 
    δ = rand(1:mcmc.ν)
    d = rand(0:min(δ,M_curr)) # Number of edges to delete
    log_ratio = 0.0 
    
    nz_edges = Set(findall(x->x>0, x_prop))
    # Delete edges
    for i in 1:d
        # @show edge_ind 
        edge = rand(edge_ind)
        log_ratio += log(length(edge_ind))
        x_prop[edge] -= 1
        if x_prop[edge] == 0 
            delete!(nz_edges, edge)
        end 
    end 
    # Add edges 
    # println("Additons:", δ-d)
    N = length(x_curr)
    for i in 1:(δ-d)
        # @show edge_ind
        edge = rand(1:N)
        if x_prop[edge] == 0 
            push!(nz_edges, edge)
        end 
        x_prop[edge] += 1
        log_ratio -= log(length(edge_ind))
    end
    # Now add term for insertions 
    
    log_ratio += (δ - 2d)*log(N)     # log((1/N)^(d-a)) = (M_prop-M_curr)*log(N) (n.b. M_curr = M_prop - d + a)
    M_prop = sum(x_prop)
    log_ratio += log(min(M_curr, δ)+1) - log(min(M_prop, δ)+1)
    # With log_α denoting the log acceptance probability and
    #       f(x) = exp(- γ d(x, xᵐ))
    # denoting the un-normalised probability of x, we have 
    # log_α = f(x_prop) - f(x_curr) + log_ratio 
    # thus the following...
    γ, d = (model.γ, model.d)
    # println(x_curr==x_prop)
    log_model = γ * (d(x_curr, x_mode) - d(x_prop, x_mode))
    log_α =  log_model + log_ratio
    # @show log_model, log_ratio, M_curr, M_prop
    if log(rand()) < log_α 
        copy!(x_curr, x_prop)
        return true 
    else 
        copy!(x_prop, x_curr)
        return false
    end 
end 

function draw_sample!(
    out::Vector{Vector{Int}},
    mcmc::SnfMcmcInsertDelete, 
    model::MultigraphSNF;
    burn_in::Int=1000, lag::Int=10,
    init::Vector{Int}=adj_mat_to_vec(model.mode, directed=model.directed, self_loops=model.self_loops)
    )

    x_curr = copy(init)
    x_prop = copy(x_curr)
    x_mode = adj_mat_to_vec(
        model.mode, 
        directed=model.directed, 
        self_loops=model.self_loops
    )
    sample_count = 1
    accept_count = 0
    iter_count = 0
    while sample_count ≤ length(out)
        iter_count += 1
        # Store value if beyond burn-in and coherent with lags
        if (iter_count > burn_in) & (((iter_count-1) % lag)==0)
            @inbounds copy!(out[sample_count], x_curr)
            sample_count += 1
        end 
        was_accepted = accept_reject!(
            x_curr, x_prop,
            mcmc, model,
            x_mode
        )
        accept_count += was_accepted

    end 
    return accept_count / iter_count
end 

function draw_sample(
    mcmc::T, 
    model::MultigraphSNF;
    desired_samples::Int=1000,
    burn_in::Int=1000, lag::Int=10,
    init::Union{Vector{Int},Matrix{Int}}=model.mode
    ) where {T<:SnfMcmcSampler}

    # We first determine whether for this metric we have a vectorised implementation 
    vectorised = is_vectorisable(model)

    if vectorised
        if typeof(init)==Matrix{Int}
            init_vec = adj_mat_to_vec(
                init, 
                directed=model.directed, 
                self_loops=model.self_loops
            )
        else 
            init_vec = init
        end 
        out = [Int[] for i in 1:desired_samples]
        accept_prob = draw_sample!(
            out, mcmc, model,
            burn_in=burn_in, lag=lag,
            init=init_vec
        )
        # Now convert back to matrices 
        directed, self_loops = (model.directed, model.self_loops)
        out = [vec_to_adj_mat(x, directed=directed, self_loops=self_loops) for x in out]
        return out, accept_prob

    else 
        # To implement 
        error("Non-vectorised scheme not yet implemented")
    end 

end 

function (mcmc::T where {T<:SnfMcmcSampler})(
    model::SNF; 
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Union{Vector{Int},Matrix{Int}}=model.mode
    )

    out, a = draw_sample(
        mcmc, 
        model, 
        desired_samples=desired_samples,
        burn_in=burn_in, 
        lag=lag,
        init=init    
    )

    p_measures = Dict(
            "Acceptance probability" => a
    )

    output = SnfMcmcOutput(
            model, 
            out, 
            p_measures
    )
    return output 

end 