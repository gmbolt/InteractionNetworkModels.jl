using Distances, RecipesBase
export CerPosterior, CerPosteriorMcmc, CerPosteriorMcmcOutput
export McmcOutputParameters, log_posterior_prob

struct CerPosterior
    data::Vector{Matrix{Bool}}
    G_prior::CER 
    α_prior::ScaledBeta
    sample_size::Int
    function CerPosterior(
        data::Vector{Matrix{Bool}},
        G_prior::CER, 
        α_prior::ScaledBeta
        ) 
        new(data,G_prior,α_prior,length(data))
    end
end 


function CerPosterior(
    data::Vector{Matrix{Int}},
    G_prior::CER, 
    α_prior::ScaledBeta
    ) 

    if prod(y->prod(x->x∈[0,1], y), data)
        data_bool = map(x->convert(Matrix{Bool},x), data)
        CerPosterior(data_bool, G_prior, α_prior)
    else 
        error("Entries must be boolean or 0/1 integers.")
    end 
end 

struct McmcOutputParameters 
    desired_samples::Int 
    burn_in::Int 
    lag::Int 
end 

struct CerPosteriorMcmc
    τ::Float64
    ε::Float64
    output::McmcOutputParameters
    counts::Vector{Int}
    function CerPosteriorMcmc(
        ;τ::Float64=0.05, ε::Float64=0.1,
        desired_samples::Int=1000, burn_in::Int=0, lag::Int=1
        )
        output = McmcOutputParameters(desired_samples, burn_in, lag)
        new(τ, ε, output, [0,0,0])
    end 
end     

function reset_counts!(mcmc::CerPosteriorMcmc)
    mcmc.counts .= 0
end 

acceptance_prob(mcmc::CerPosteriorMcmc) = (
    mcmc.counts[1]/mcmc.counts[3],
    mcmc.counts[2]/mcmc.counts[3] 
)

struct CerPosteriorMcmcOutput 
    G_sample::Vector{Matrix{Bool}}
    α_sample::Vector{Float64}
    posterior::CerPosterior
end 

# get_sample_matrices(out::CerPosteriorMcmcOutput) = vec_to_adj_mat.(
#     out.G_sample, 
#     directed=out.posterior.G_prior.directed,
#     self_loops=out.posterior.G_prior.self_loops
# )

@recipe function f(
    output::CerPosteriorMcmcOutput,
    G_true::Matrix{Bool}
    ) 
    d = Hamming()
    G_sample = output.G_sample # Observations as vectors 
    α_sample = output.α_sample
    layout := (2,1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "α"]
    size --> (800, 600)
    margin --> 5mm
    y1 = [d(G_true,x) for x in G_sample]
    y2 = α_sample 
    hcat(y1,y2)
end 

function accept_reject_mode_fast!(
    x_curr::Vector{Bool},
    x_prop::Vector{Bool},
    α_curr::Float64,
    posterior::CerPosterior,
    x_data_sum::Vector{Int},
    x_0::Vector{Bool},
    τ::Float64
    )

    n = length(posterior.data)
    α₀ = posterior.G_prior.α
    
    suff_stat_diff = 0.0 # C′-C
    for (i,v) in enumerate(x_prop)
        if rand() < τ
            c = 2*x_data_sum[i] - n
            suff_stat_diff += v ? c : -c 
            x_prop[i] = !(x_prop[i]) # Flip edge 
        end 
    end 
    d = Hamming()
    prior_diffs = d(x_prop, x_0) - d(x_curr, x_0)

    log_α = (
        suff_stat_diff * log(α_curr) - suff_stat_diff * log(1-α_curr)
        + + prior_diffs * log(α₀) - prior_diffs * log(1-α₀)
    )
    if log(rand()) < log_α 
        copy!(x_curr, x_prop)
        return true
    else 
        copy!(x_prop, x_curr)
        return false 
    end 
end 


function accept_reject_mode!(
    x_curr::Vector{Bool},
    x_prop::Vector{Bool},
    α_curr::Float64,
    posterior::CerPosterior,
    x_data_sum::Vector{Int},
    x_0::Vector{Bool},
    τ::Float64
    )

    n = length(x_curr)
    α₀ = posterior.G_prior.α
    
    log_α = 0.0 
    for i in eachindex(x_prop)
        if rand() < τ
            x_prop[i] = !(x_prop[i]) # Flip edge 
        end 
    end 
    data = posterior.data
    d = Hamming()
    suff_stat_curr = sum(d(x_curr,x) for x in data)
    suff_stat_prop = sum(d(x_prop,x) for x in data)
    prior_diffs = d(x_prop, x_0) - d(x_curr, x_0)
    log_α += (
        (suff_stat_prop - suff_stat_curr) * log(α_curr)
        + (suff_stat_curr - suff_stat_prop) * log(1-α_curr)
        + prior_diffs * log(α₀) - prior_diffs * log(1-α₀)
    )

    if log(rand()) < log_α 
        copy!(x_curr, x_prop)
        return true
    else 
        copy!(x_prop, x_curr)
        return false 
    end 
end 

function accept_reject_alpha(
    α_curr::Float64,
    x_curr::Vector{Bool},
    posterior::CerPosterior,
    ε::Float64
    )

    n = length(x_curr)
    a,b = (posterior.α_prior.α, posterior.α_prior.β)
    data = posterior.data
    tot_ham_dist = sum(
        sum(u!==v for (u,v) in zip(x_curr,x)) for x in data
    )
    α_prop = rand_reflect(α_curr, ε, 0.0, 0.5)
    log_α = (
        tot_ham_dist * (log(α_prop)-log(α_curr)) 
        + (length(data)*n - tot_ham_dist) * (log(1-α_prop)-log(1-α_curr))
        + (a-1) * (log(α_prop)-log(α_curr)) 
        + (b-1) * (log(1 - 2α_prop) - log(1-2α_curr))
    )

    if log(rand()) < log_α 
        return α_prop, true
    else 
        return α_curr, false
    end 
end 

function draw_sample!(
    sample_out_G::T, 
    sample_out_alpha::S, 
    mcmc::CerPosteriorMcmc,
    posterior::CerPosterior;
    burn_in::Int=mcmc.output.burn_in,
    lag::Int=mcmc.output.lag,
    G_init::Union{Matrix{Int},Matrix{Bool}}=rand(Bool, posterior.G_prior.mode),
    α_init::Float64=0.25,
    loading_bar::Bool=true
    ) where {T<:Union{Vector{Vector{Bool}},SubArray},S<:Union{Vector{Float64},SubArray}}

    if loading_bar
        iter = Progress(
            length(sample_out_G)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (joint)....")  # Loading bar. Minimum update interval: 1 second
    end 
    G_prior = posterior.G_prior
    is_directed = G_prior.directed
    has_self_loops = G_prior.self_loops
    x_0 = adj_mat_to_vec(
        G_prior.mode, 
        directed=is_directed, 
        self_loops=has_self_loops
    )
    data = posterior.data 
    x_data_sum = adj_mat_to_vec(
        sum(data),
        directed=is_directed, 
        self_loops=has_self_loops
    )
    x_curr = adj_mat_to_vec(
        G_init, 
        directed=is_directed, 
        self_loops=has_self_loops
    )
    x_prop = copy(x_curr)
    α_curr = α_init 

    reset_counts!(mcmc)
    G_acount = 0 
    α_acount = 0 
    sample_count = 1
    i = 0 
    while sample_count ≤ length(sample_out_G)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out_G[sample_count] = copy(x_curr)
            @inbounds sample_out_alpha[sample_count] = α_curr 
            sample_count += 1
        end 
        # Accept-reject move
        G_acount += accept_reject_mode_fast!(
            x_curr, x_prop, 
            α_curr, posterior,
            x_data_sum, 
            x_0,
            mcmc.τ
        )

        # Accept-reject dispersion 
        α_curr, was_acc = accept_reject_alpha(
            α_curr, x_curr,
            posterior, 
            mcmc.ε            
        )
        α_acount += was_acc

        if loading_bar
            next!(iter)
        end 
    end 
    mcmc.counts[1:3] = [G_acount,α_acount, i]
end 

function draw_sample(
    mcmc::CerPosteriorMcmc,
    posterior::CerPosterior;
    desired_samples::Int=mcmc.output.desired_samples, 
    args...
    ) 
    sample_out_G = Vector{Vector{Bool}}(undef, desired_samples)
    sample_out_alpha = zeros(desired_samples)
    draw_sample!(
        sample_out_G, 
        sample_out_alpha, 
        mcmc, 
        posterior;
        args...
    )
    return sample_out_G, sample_out_alpha
end 

function cer_post_log_prob(
    x::Vector{Bool}, 
    α::Float64,
    data_vec::Vector{Vector{Bool}},
    x₀::Vector{Bool},
    α₀::Float64,
    α_prior::UnivariateDistribution
    )

    n = length(data_vec)
    E = length(x)

    d = Hamming()
    S = sum(d(y,x) for y in data_vec)

    return (
        S * log(α) + (n*E -  S) * log(1-α)
        + d(x,x₀) * log(α₀) + (E - d(x,x₀)) * log(1-α₀)
        + log(pdf(α_prior, α))
    )
end 

function (mcmc::CerPosteriorMcmc)(
    posterior::CerPosterior;
    args...
    )

    G_sample, α_sample = draw_sample(
        mcmc, posterior;
        args...
    )
    # Make sample into matrices 
    G_sample = vec_to_adj_mat.(
        G_sample, 
        directed=posterior.G_prior.directed, 
        self_loops=posterior.G_prior.self_loops
    )
    return CerPosteriorMcmcOutput(
        G_sample, α_sample,
        posterior
    )
    
end 

function log_posterior_prob(
    out::CerPosteriorMcmcOutput
    )
    posterior = out.posterior
    dir, sl = (
        posterior.G_prior.directed,
        posterior.G_prior.self_loops
    )
    x_sample = adj_mat_to_vec.(
        out.G_sample, 
        directed=dir, 
        self_loops=sl
    )
    # Post-hoc log-probability evaluation
    data_vec = adj_mat_to_vec.(
        posterior.data,
        directed=dir, 
        self_loops=sl
    )
    x₀ = adj_mat_to_vec(
        posterior.G_prior.mode, 
        directed=dir, 
        self_loops=sl
    )
    α₀ = posterior.G_prior.α
    log_post_prob = [ 
        cer_post_log_prob(
            x, α, 
            data_vec, 
            x₀, α₀, 
            posterior.α_prior
        ) for (x,α) in zip(x_sample, out.α_sample)
    ]
    return log_post_prob
end 