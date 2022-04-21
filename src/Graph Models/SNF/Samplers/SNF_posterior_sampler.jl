using RecipesBase 

export SnfPosteriorSampler, SnfPosteriorMcmcOutput
export get_sample_matrices

struct SnfAuxiliaryMcmc{T,N,S<:McmcSampler}
    mcmc::S  # Mcmc move (leave general since will also use for graph models)
    data::Vector{Array{T,N}} # To store auxiliary samples 
    init_at_prev::Bool 
end 

# We need to specify (i) if vectorised (ii) element types. 
# (Both will will change how the auxiliary data is sampled)
struct SnfPosteriorSampler{T<:Union{Int,Bool},N,S₁<:McmcMove,S₂<:McmcSampler}
    move::S₁    # Move for mode update 
    ε::Float64 # Neighborhood for gamma proposal
    aux::SnfAuxiliaryMcmc{T,N,S₂}  # auxiliary MCMC sampler
    output::McmcOutputParameters
    suff_stats::Vector{Float64}
    γ_counts::Vector{Int}
    function SnfPosteriorSampler(
        move::Q, aux_mcmc::McmcSampler,
        posterior::SnfPosterior{T,N,V,S};
        ε::Float64=0.15, 
        desired_samples::Int=1000, burn_in::Int=0, lag::Int=1,
        aux_init_at_prev::Bool=false,
        ) where {Q<:McmcMove,T,N,V,S}
        suff_stats = [0.0,0.0]
        aux_data = Array{T,N}[]
        γ_counts = [0,0]
        aux = SnfAuxiliaryMcmc(aux_mcmc, aux_data, aux_init_at_prev)
        output = McmcOutputParameters(desired_samples, burn_in, lag)
        new{T,N,Q,typeof(aux_mcmc)}(
            move, ε, aux,
            output,
            suff_stats, 
            γ_counts
            )
    end 
end 

Base.show(io::IO, x::SnfPosteriorSampler) = print(io, typeof(x))

struct SnfPosteriorMcmcOutput{T,N,V,S} 
    x_sample::Vector{Array{T,N}}
    γ_sample::Vector{Float64}
    suff_stat_trace::Vector{Float64}
    posterior::SnfPosterior{T,N,V,S}
end 

Base.show(io::IO, x::SnfPosteriorMcmcOutput{T,N,V,S}) where {T,N,V,S} = print(io, typeof(x))

@recipe function f(
    output::SnfPosteriorMcmcOutput{T,N,V,S},
    x_true::Array{T,N}
    ) where {T,N,V,S}

    x_sample = output.x_sample
    γ_sample = output.γ_sample
    layout := (2,1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "γ"]
    size --> (800, 600)
    margin --> 5mm
    y1 = map(x->output.posterior.d(x_true,x), x_sample)
    y2 = γ_sample 
    hcat(y1,y2)
end 

function get_sample_matrices(
    output::SnfPosteriorMcmcOutput{T,N,V,S}
    ) where {T,N,V,S}
    if N == 1
        dir, sl = (
            output.posterior.directed,
            output.posterior.self_loops
        )
        return vec_to_adj_mat.(
            output.x_sample,
            directed=dir, 
            self_loops=sl
        )
    elseif N==2  
        return output.x_sample
        
    else 
        error("Data of invalid dimension, should be 1d or 2d.")
    end 
end 


function eval_accept_prob(
    x_curr::Array{T,N},
    x_prop::Array{T,N},
    γ_curr::Float64,
    aux::SnfAuxiliaryMcmc{T,N,S},
    posterior::SnfPosterior{T,N,V,R},
    log_ratio::Float64,
    suff_stats::Vector{Float64}
    ) where {T,N,S,V,R}

    G_prior = posterior.G_prior
    d = posterior.d
    x_mode_prior, γ_prior = (
        G_prior.mode, G_prior.γ
    )

    # Infer aux model from mode prior 
    aux_model = similar(
        G_prior, 
        x_prop, 
        γ_curr
    )

    if aux.init_at_prev
        @inbounds tmp = deepcopy(aux.data[end])
        draw_sample!(aux.data, aux.mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux.data, aux.mcmc, aux_model)
    end 

    aux_log_lik_ratio = -γ_curr * (
        sum(d(x, x_curr) for x in aux.data)
        - sum(d(x, x_prop) for x in aux.data)
    )

    suff_stats[2] = sum(d(x, x_prop) for x in posterior.data)

    log_lik_ratio = -γ_curr * (
        suff_stats[2] - suff_stats[1] # (prop-curr)
    )

    log_prior_ratio = -γ_prior * (
        d(x_prop, x_mode_prior) - d(x_curr, x_mode_prior)
    )

    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    return log_α

end 

function accept_reject_mode!(
    x_curr::Array{T,N},
    x_prop::Array{T,N},
    γ_curr::Float64,
    move::McmcMove,
    aux::SnfAuxiliaryMcmc{T,N,S},
    posterior::SnfPosterior{T,N,V,R},
    suff_stats::Vector{Float64}
    ) where {T,N,S,V,R}

    # Do imcmc proposal 
    log_ratio = prop_sample!(x_curr, x_prop, move)

    log_α = eval_accept_prob(
        x_curr, x_prop, γ_curr, 
        aux,
        posterior, 
        log_ratio,
        suff_stats
    )

    move.counts[2] += 1
    
    if log(rand()) < log_α
        # We accept! 
        move.counts[1] += 1
        suff_stats[1] = suff_stats[2]  # Update sufficient stat
        copy!(x_curr, x_prop)
    else 
        # We reject!
        copy!(x_prop, x_curr)
    end 
end 

# Must specialise for Gibbs scan move (for vector)
function accept_reject_mode!(
    x_curr::Array{T,1},
    x_prop::Array{T,1},
    γ_curr::Float64,
    move::GibbsScanMove,
    aux::SnfAuxiliaryMcmc{T,1,S},
    posterior::SnfPosterior{T,1,V,R},
    suff_stats::Vector{Float64}
    ) where {T,S,V,R}

    move.counts[2] += length(x_curr)
    for i in eachindex(x_curr)
        log_ratio = prop_sample_entry!(x_curr, x_prop, move, i)
        log_α = eval_accept_prob(
            x_curr, x_prop, γ_curr, 
            aux,
            posterior, 
            log_ratio, 
            suff_stats
        )
        
        if log(rand()) < log_α
            # We accept! 
            move.counts[1] += 1
            suff_stats[1] = suff_stats[2]  # Update sufficient stat
            copy!(x_curr, x_prop)
        else 
            # We reject!
            copy!(x_prop, x_curr)
        end 
    end 
end 

# Wrapper to unpack the mcmc bits and pass...
function accept_reject_mode!(
    x_curr::Array{T,N},
    x_prop::Array{T,N},
    γ_curr::Float64,
    mcmc::SnfPosteriorSampler{T,N,S₁,S₂},
    posterior::SnfPosterior{T,N,V,R}
    ) where {T,N,S₁,S₂,V,R}

    accept_reject_mode!(
        x_curr, x_prop, γ_curr,
        mcmc.move, 
        mcmc.aux,
        posterior,
        mcmc.suff_stats
    )
end 

function initialise!(
    aux::SnfAuxiliaryMcmc{T,N,S},
    x_init::Array{T,N},
    γ_init::Float64,
    G_prior::SNF,
    n::Int # Sample size
    ) where {T,N,S<:McmcSampler}

    if length(aux.data) ≥ n
        resize!(aux.data, n)
    else
        append!(aux.data, [similar(x_init) for i in 1:(n-length(aux.data))])
    end 
    aux_model = similar(
        G_prior, 
        x_init, 
        γ_init
    )
    draw_sample!(aux.data, aux.mcmc, aux_model)

end 

function draw_sample_mode!(
    sample_out::Union{Vector{Vector{Int}}, SubArray},
    mcmc::SnfPosteriorSampler{Int,1,S₁,S₂},
    posterior::VecMultigraphSnfPosterior,
    γ_fixed::Float64;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    G_init::Union{Matrix{Int},Vector{Int}}=sample_frechet_mean(posterior.data_vec, posterior.d)[1],
    loading_bar::Bool=true
    ) where {S₁,S₂}

    if loading_bar
        iter = Progress(
            length(sample_out)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 
    
    # Initialise states 
    if typeof(G_init)==Vector{Int}
        x_curr = copy(G_init)
        x_prop = copy(x_curr)
    elseif typeof(G_init)==Matrix{Int}
        directed, self_loops = (
            posterior.directed, 
            posterior.self_loops
        )
        x_curr = adj_mat_to_vec(
            G_init,
            directed=directed, 
            self_loops=self_loops
        )
        x_prop = copy(x_curr)
    else 
        error("Initial value of unsupported type.")
    end 

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 
    reset_counts!(mcmc.move) # Reset counts for mcmc move (tracks acceptances) 

    # Initialise aux mcmc (resize data and draw initial sample)
    initialise!(mcmc.aux, x_curr, γ_curr, posterior.G_prior, posterior.sample_size)

    # Initialise sufficient statistic 
    d, data = (posterior.d, posterior.data)
    suff_stats = mcmc.suff_stats
    suff_stats[1] = sum(d(x_curr, x) for x in data)

    # Storage for sample statistic 
    sample_suff_stat = zeros(length(sample_out))

    while sample_count ≤ length(sample_out)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out[sample_count] = deepcopy(x_curr)
            @inbounds sample_suff_stat[sample_count] = suff_stats[1]
            sample_count += 1
        end 
        accept_reject_mode!(
            x_curr, x_prop, 
            γ_fixed, 
            mcmc, posterior
        )
        # println(S_prop)
        if loading_bar
            next!(iter)
        end 
    end 
    return sample_suff_stat
end 

function draw_sample_mode(
    mcmc::SnfPosteriorSampler,
    posterior::VecMultigraphSnfPosterior,
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    args...
    )

    sample_out = [Int[] for i in 1:desired_samples]
    sample_suff_stat = draw_sample_mode!(
        sample_out, 
        mcmc, posterior, γ_fixed,
        args...
    )
    
    return sample_out, sample_suff_stat

end 

function (mcmc::SnfPosteriorSampler)(
    posterior::VecMultigraphSnfPosterior, 
    γ_fixed::Float64;
    args...
    )

    x_sample, suff_stat_trace = draw_sample_mode(
        mcmc, posterior, γ_fixed,
        arg...
    )
    γ_sample = fill(γ_fixed, length(S_sample))

    # .... define output struct 
    return SnfPosteriorMcmcOutput(
        x_sample,
        γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 

function accept_reject_gamma(
    γ_curr::Float64,
    x_curr::Vector{Int},
    mcmc::SnfPosteriorSampler,
    posterior::VecMultigraphSnfPosterior
    )
    
    G_prior = posterior.G_prior
    d = posterior.d
    aux = mcmc.aux
    ε, suff_stats = (
        mcmc.ε,
        mcmc.suff_stats
    )
    # Draw proposal 
    γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

    aux_model = similar(
        G_prior, 
        x_curr, 
        γ_prop
    )

    if aux.init_at_prev
        @inbounds tmp = deepcopy(aux.data[end])
        draw_sample!(aux.data, aux.mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux.data, aux.mcmc, aux_model)
    end 

    # Accept reject
    log_lik_ratio = (γ_curr - γ_prop) * suff_stats[1]
    aux_log_lik_ratio = (γ_prop - γ_curr) * sum(d(x,x_curr) for x in aux.data)

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

function draw_sample_gamma!(
    sample_out::Union{Vector{Float64}, SubArray},
    mcmc::SnfPosteriorSampler,
    posterior::SnfPosterior,
    x_fixed::Vector{Int};
    burn_in::Int=mcmc.output.burn_in,
    lag::Int=mcmc.output.lag,
    γ_init::Float64=4.0,
    loading_bar::Bool=true
    )

    if loading_bar
        iter = Progress(
            length(sample_out)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 

    γ_curr = γ_init 
    x_curr = copy(x_fixed)

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 

    # Initialise aux mcmc (resize data and draw initial sample)
    initialise!(mcmc.aux, x_curr, γ_curr, posterior.G_prior, posterior.sample_size)

    acc_count = 0
    i = 0 # Counter for iterations 
    sample_count = 1  # Which sample we are working to get 

    while sample_count ≤ length(sample_out)
        i += 1
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = γ_curr
            sample_count += 1
        end 

        γ_curr, was_acc = accept_reject_gamma(
            γ_curr, x_curr, 
            mcmc, posterior 
        )
        acc_count += was_acc

        if loading_bar
            next!(iter)
        end 

    end 
    mcmc.γ_counts = [acc_count, i]
end 


function draw_sample_gamma(
    mcmc::SnfPosteriorSampler,
    posterior::SnfPosterior,
    x_fixed::Vector{Int};
    desired_samples::Int=mcmc.output.desired_samples,
    args...
    )

    sample_out = zeros(desired_samples)
    draw_sample_gamma!(
        sample_out, 
        mcmc, posterior, 
        x_fixed, 
        args...
        )
    return sample_out

end 


function (mcmc::SnfPosteriorSampler)(
    posterior::SnfPosterior, 
    x_fixed::Vector{Int};
    args...
    ) 

    γ_sample = draw_sample_gamma(
        mcmc, posterior, γ_fixed,
        args...
    )
    x_sample = [copy(x_fixed) for i in eachindex(γ_sample)]
    suff_stat = sum(posterior.di(x,x_fixed) for x in posterior.data)
    suff_stat_trace = fill(suff_stat, length(γ_sample))
    return SnfPosteriorMcmcOutput(
        x_sample, γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 

function draw_sample!(
    sample_out_x::T,
    sample_out_gamma::S,
    mcmc::SnfPosteriorSampler,
    posterior::SnfPosterior;
    burn_in::Int=mcmc.output.burn_in,
    lag::Int=mcmc.output.lag,
    G_init::Union{Matrix{Int},Vector{Int}}=sample_frechet_mean(posterior.data, posterior.d)[1],
    γ_init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{Vector{Vector{Int}},SubArray},S<:Union{Vector{Float64},SubArray}}

    if loading_bar
        iter = Progress(
            length(sample_out_x)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (joint)....")  # Loading bar. Minimum update interval: 1 second
    end 

    # Initialise states 
    if typeof(G_init)==Vector{Int}
        x_curr = copy(G_init)
        x_prop = copy(x_curr)
    elseif typeof(G_init)==Matrix{Int}
        directed, self_loops = (
            posterior.directed, 
            posterior.self_loops
        )
        x_curr = adj_mat_to_vec(
            G_init,
            directed=directed, 
            self_loops=self_loops
        )
        x_prop = copy(x_curr)
    else 
        error("Initial value of unsupported type.")
    end 
    # @show x_curr
    γ_curr = γ_init 

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 
    γ_acc_count = 0
    reset_counts!(mcmc.move) # Reset counts for mcmc move (tracks acceptances)

    # Initialise aux mcmc (resize data and draw initial sample)

    initialise!(mcmc.aux, x_curr, γ_curr, posterior.G_prior, posterior.sample_size)

    # Initialise sufficient statistic 
    d, data = (posterior.d, posterior.data)
    suff_stats = mcmc.suff_stats
    suff_stats[1] = sum(d(x_curr, x) for x in data)

    # Storage for sample statistic 
    sample_suff_stat = zeros(length(sample_out_x))

    while sample_count ≤ length(sample_out_x)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out_x[sample_count] = copy(x_curr)
            @inbounds sample_out_gamma[sample_count] = γ_curr 
            @inbounds sample_suff_stat[sample_count] = suff_stats[1]
            sample_count += 1
        end 
        # Accept-reject move

        accept_reject_mode!(
            x_curr, x_prop, 
            γ_curr, 
            mcmc, posterior
        )
        
        # Accept-reject dispersion 
        γ_curr, was_acc = accept_reject_gamma(
            γ_curr, x_curr, 
            mcmc, 
            posterior
        )

        γ_acc_count += was_acc
        # println(S_prop)
        if loading_bar
            next!(iter)
        end 
    end 
    mcmc.γ_counts[1:2] = [γ_acc_count, i]
    return sample_suff_stat
end 

function draw_sample(
    mcmc::SnfPosteriorSampler,
    posterior::SnfPosterior;
    desired_samples::Int=mcmc.output.desired_samples,
    args...
    ) 

    sample_out_x = Vector{Vector{Int}}(undef, desired_samples)
    sample_out_gamma = zeros(desired_samples)
    suff_stat_sample = draw_sample!(
        sample_out_x,
        sample_out_gamma, 
        mcmc, 
        posterior;
        args...
        )
    return sample_out_x, sample_out_gamma, suff_stat_sample
end 

function (mcmc::SnfPosteriorSampler)(
    posterior::SnfPosterior;
    args...
    ) 
    x_sample, γ_sample, suff_stat_trace = draw_sample(
        mcmc, posterior;
        args...
    )
    return SnfPosteriorMcmcOutput(
        x_sample, γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 
