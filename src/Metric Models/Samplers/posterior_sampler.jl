using RecipesBase, Measures
export IexMcmcSampler, AuxiliaryMcmc, PosteriorMcmcOutput
export eval_accept_prob

struct AuxiliaryMcmc{S,T}
    mcmc::S  # Mcmc move (leave general since will also use for graph models)
    data::Vector{T} # To store auxiliary samples 
    init_at_prev::Bool 
end 

struct IexMcmcSampler{T<:InvMcmcMove,S<:AuxiliaryMcmc}
    move::T    # Move for mode update 
    ε::Float64 # Neighborhood for gamma proposal
    aux::S
    desired_samples::Int 
    burn_in::Int
    lag::Int
    K::Int
    pointers::InteractionSequence{Int}
    suff_stats::Vector{Float64}
    γ_counts::Vector{Int}
    function IexMcmcSampler(
        move::T, aux_mcmc::InvMcmcSampler;
        ε::Float64=0.15, 
        desired_samples::Int=1000, burn_in::Int=0, lag::Int=1,
        K::Int=100, aux_init_at_prev::Bool=false
        ) where {T<:InvMcmcMove}
        pointers = [Int[] for i in 1:(2K)]
        suff_stats = [0.0,0.0]
        aux_data = [Vector{Int}[]]
        γ_counts = [0,0]
        aux = AuxiliaryMcmc(aux_mcmc, aux_data, aux_init_at_prev)
        new{T,typeof(aux)}(
            move, ε, aux,
            desired_samples, burn_in, lag, 
            K, pointers,
            suff_stats, 
            γ_counts
            )
    end 
end 

Base.show(io::IO, x::IexMcmcSampler) = print(io, typeof(x))


struct PosteriorMcmcOutput{T<:Union{SisPosterior,SimPosterior}} 
    S_sample::Vector{InteractionSequence{Int}}
    γ_sample::Vector{Float64}
    suff_stat_trace::Vector{Float64}
    posterior::T
end 

Base.show(io::IO, x::PosteriorMcmcOutput) = print(io, typeof(x))


@recipe function f(
    output::PosteriorMcmcOutput,
    S_true::InteractionSequence{Int}
    ) 

    S_sample = output.S_sample
    γ_sample = output.γ_sample
    layout := (2,1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "γ"]
    size --> (800, 600)
    margin --> 5mm
    y1 = map(x->output.posterior.dist(S_true,x), S_sample)
    y2 = γ_sample 
    hcat(y1,y2)
end 

acceptance_prob(mcmc::IexMcmcSampler) = (mode=acceptance_prob(mcmc.move),γ=mcmc.γ_counts[1]/mcmc.γ_counts[2])

function eval_accept_prob(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    γ_curr::Float64,
    aux::AuxiliaryMcmc,
    posterior::T,
    log_ratio::Float64,
    suff_stats::Vector{Float64}
    ) where {T<:Union{SisPosterior,SimPosterior}}

    dist, V, K_inner, K_outer = (
        posterior.dist, posterior.V, 
        posterior.K_inner, posterior.K_outer
    )
    mode_prior, γ_prior = (
        posterior.S_prior.mode, posterior.S_prior.γ
    )

    # Infer aux model from mode prior 
    aux_model = typeof(posterior.S_prior)(
        S_prop, γ_curr, 
        dist, V, 
        K_inner, K_outer
    )

    if aux.init_at_prev
        @inbounds tmp = deepcopy(aux.data[end])
        draw_sample!(aux.data, aux.mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux.data, aux.mcmc, aux_model)
    end 

    aux_log_lik_ratio = -γ_curr * (
        sum(dist(x, S_curr) for x in aux.data)
        - sum(dist(x, S_prop) for x in aux.data)
    )
    # @show aux_log_lik_ratio
    
    suff_stats[2] = sum(dist(x, S_prop) for x in posterior.data)

    log_lik_ratio = -γ_curr * (
        suff_stats[2] - suff_stats[1] # (prop-curr)
    )

    log_prior_ratio = -γ_prior * (
        dist(S_prop, mode_prior) - dist(S_curr, mode_prior)
    )

    log_α = log_lik_ratio + log_prior_ratio + aux_log_lik_ratio + log_ratio 

    # Log acceptance probability
    if typeof(posterior)==SisPosterior 
        return log_α
    else 
        log_multinom_term = log_multinomial_ratio(S_curr, S_prop)
        return log_α + log_multinom_term 
    end 

end 

function accept_reject_mode!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    γ_curr::Float64,
    pointers::InteractionSequence{Int},
    move::InvMcmcMove,
    aux::AuxiliaryMcmc,
    posterior::S,
    suff_stats::Vector{Float64}
    ) where {S<:Union{SisPosterior,SimPosterior}}

    move.counts[2] += 1
    
    # Do imcmc proposal (log_ratio gets passed to eval_accept_prob()...)
    log_ratio = prop_sample!(S_curr, S_prop, move, pointers, posterior.V)

    # Adjust for dimension bounds (reject if outside of them)
    if any(!(1 ≤ length(x) ≤ posterior.K_inner.u) for x in S_prop)
        log_α = -Inf
    elseif !(1 ≤ length(S_prop) ≤ posterior.K_outer.u)
        log_α = -Inf
    else 
        log_α = eval_accept_prob(
            S_curr, S_prop, γ_curr, 
            aux,
            posterior, 
            log_ratio,
            suff_stats
        )
    end 

    if log(rand()) < log_α
        # We accept! 
        move.counts[1] += 1
        suff_stats[1] = suff_stats[2]  # Update sufficient stat
        enact_accept!(S_curr, S_prop, pointers, move)
    else 
        # We reject!
        enact_reject!(S_curr, S_prop, pointers, move)
    end 
    
end 

function accept_reject_mode!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    γ_curr::Float64,
    pointers::InteractionSequence{Int},
    mix_move::InvMcmcMixtureMove,
    aux::AuxiliaryMcmc,
    posterior::T,
    suff_stats::Vector{Float64}
    ) where {T<:Union{SisPosterior,SimPosterior}}


    # Sample move 
    p = mix_move.p  # Mixture proportions 
    β = rand()      # Random unif(0,1)
    z, i = (0.0, 0) # To store cumulative prob and index 
    for prob in p
        if z > β 
            break 
        end 
        i += 1
        z += prob
    end 

    # Select ith move
    move = mix_move.moves[i]
    # Do accept-reject for the move
    accept_reject_mode!(
        S_curr, S_prop, γ_curr, 
        pointers,
        move,  
        aux,
        posterior,
        suff_stats
    )
end 

function accept_reject_mode!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    γ_curr::Float64,
    pointers::InteractionSequence{Int},
    mcmc::IexMcmcSampler,
    posterior::T
    ) where {T<:Union{SisPosterior,SimPosterior}}
    accept_reject_mode!(
        S_curr, S_prop, γ_curr,
        pointers, 
        mcmc.move, 
        mcmc.aux,
        posterior,
        mcmc.suff_stats
    )
end 

function initialise!(
    aux::AuxiliaryMcmc{S,T},
    S_init::T, # Initial value for aux chain
    γ_init::Float64, # Used to set-up auxiliary model 
    S_prior::V,
    n::Int # Sample size
    ) where {T,S,V<:Union{SIS,SIM}}

    if length(aux.data) ≥ n
        resize!(aux.data, n)
    else
        append!(aux.data, [similar(S_init) for i in 1:(n-length(aux.data))])
    end 
    aux_model = similar(
        S_prior, 
        S_init, 
        γ_init
    )
    draw_sample!(aux.data, aux.mcmc, aux_model)

end 

function draw_sample_mode!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::IexMcmcSampler,
    posterior::T,
    γ_fixed::Float64;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist)[1],
    loading_bar::Bool=true
    ) where {T<:Union{SisPosterior,SimPosterior}}

    if loading_bar
        iter = Progress(
            length(sample_out)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 

    pointers = mcmc.pointers
    S_curr, S_prop = intialise_states!(pointers, S_init)

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 
    reset_counts!(mcmc.move) # Reset counts for mcmc move (tracks acceptances) 

    # Initialise aux data 
    initialise!(mcmc.aux, S_curr, γ_fixed, posterior.S_prior, posterior.sample_size)

    # Initialise sufficient statistic 
    dist, data = (posterior.dist, posterior.data)
    suff_stats = mcmc.suff_stats
    suff_stats[1] = sum(dist(S_curr, x) for x in data)

    # Storage for sample statistic 
    sample_suff_stat = zeros(length(sample_out))

    while sample_count ≤ length(sample_out)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out[sample_count] = deepcopy(S_curr)
            @inbounds sample_suff_stat[sample_count] = suff_stats[1]
            sample_count += 1
        end 
        accept_reject_mode!(
            S_curr, S_prop, 
            γ_fixed, pointers, 
            mcmc, posterior
        )
        # println(S_prop)
        if loading_bar
            next!(iter)
        end 
    end 
    return_states!(S_curr, S_prop, pointers)
    return sample_suff_stat
end 

function draw_sample_mode(
    mcmc::IexMcmcSampler,
    posterior::T,
    γ_fixed::Float64;
    desired_samples::Int=mcmc.desired_samples,
    args...
    ) where {T<:Union{SisPosterior,SimPosterior}}

    sample_out = [[Int[]] for i in 1:desired_samples]
    sample_suff_stat = draw_sample_mode!(
        sample_out, 
        mcmc, posterior, γ_fixed;
        args...
    )
    
    return sample_out, sample_suff_stat

end 

function (mcmc::IexMcmcSampler)(
    posterior::T, 
    γ_fixed::Float64;
    args...
    ) where {T<:Union{SisPosterior,SimPosterior}}

    S_sample, suff_stat_trace = draw_sample_mode(
        mcmc, posterior, γ_fixed;
        args...
    )
    γ_sample = fill(γ_fixed, length(S_sample))

    return PosteriorMcmcOutput(
        S_sample, γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 



function accept_reject_gamma(
    γ_curr::Float64,
    S_curr::InteractionSequence{Int},
    mcmc::IexMcmcSampler,
    posterior::T
    ) where {T<:Union{SisPosterior,SimPosterior}}

    dist, V, K_inner, K_outer = (
        posterior.dist, posterior.V, 
        posterior.K_inner, posterior.K_outer
    )
    aux = mcmc.aux
    ε, suff_stats = (
        mcmc.ε,
        mcmc.suff_stats
    )

    γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

    aux_model = typeof(posterior.S_prior)(
        S_curr, γ_prop, 
        dist, V, 
        K_inner, K_outer
    )

    if aux.init_at_prev
        @inbounds tmp = deepcopy(aux.data[end])
        draw_sample!(aux.data, aux.mcmc, aux_model, init=tmp)
    else 
        draw_sample!(aux.data, aux.mcmc, aux_model)
    end 

    # Accept reject
    log_lik_ratio = (γ_curr - γ_prop) * suff_stats[1]
    aux_log_lik_ratio = (γ_prop - γ_curr) * sum(dist(x,S_curr) for x in aux.data)

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
    mcmc::IexMcmcSampler,
    posterior::T,
    S_fixed::InteractionSequence{Int};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    γ_init::Float64=4.0,
    loading_bar::Bool=true
    ) where {T<:Union{SisPosterior,SimPosterior}}

    if loading_bar
        iter = Progress(
            length(sample_out)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (disperison conditional)....")  # Loading bar. Minimum update interval: 1 second
    end 

    γ_curr = γ_init 
    S_curr = deepcopy(S_fixed)

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 

    # Initialise aux data 
    initialise!(mcmc.aux, S_curr, γ_curr, posterior.S_prior, posterior.sample_size)

    # Initialise sufficient statistic 
    dist, data = (posterior.dist, posterior.data)
    suff_stats = mcmc.suff_stats
    suff_stats[1] = sum(dist(S_curr, x) for x in data)

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
            γ_curr, S_curr, 
            mcmc, posterior 
        )
        acc_count += was_acc

        if loading_bar
            next!(iter)
        end 

    end 
    mcmc.γ_counts[1:2] = [acc_count, i]
    return
end 


function draw_sample_gamma(
    mcmc::IexMcmcSampler,
    posterior::T,
    S_fixed::InteractionSequence{Int};
    desired_samples::Int=mcmc.desired_samples,
    args...
    ) where {T<:Union{SisPosterior,SimPosterior}}

    sample_out = zeros(desired_samples)
    draw_sample_gamma!(
        sample_out, 
        mcmc, posterior, 
        S_fixed;
        args...
        )
    return sample_out

end 


function (mcmc::IexMcmcSampler)(
    posterior::T, 
    S_fixed::InteractionSequence{Int};
    args...
    ) where {T<:Union{SisPosterior,SimPosterior}}

    γ_sample = draw_sample_gamma(
        mcmc, posterior, S_fixed;
        args...
    )
    S_sample = [deepcopy(S_fixed) for i in eachindex(γ_sample)]
    suff_stat = sum(posterior.dist(x,S_fixed) for x in posterior.data)
    suff_stat_trace = fill(suff_stat, length(γ_sample))
    return PosteriorMcmcOutput(
        S_sample, γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 

function draw_sample!(
    sample_out_S::T,
    sample_out_gamma::S,
    mcmc::IexMcmcSampler,
    posterior::V;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    S_init::Vector{Path{Int}}=sample_frechet_mean(posterior.data, posterior.dist)[1],
    γ_init::Float64=5.0,
    loading_bar::Bool=true
    ) where {T<:Union{InteractionSequenceSample{Int},SubArray},S<:Union{Vector{Float64},SubArray},V<:Union{SisPosterior,SimPosterior}}

    if loading_bar
        iter = Progress(
            length(sample_out_S)*lag + burn_in, # How many iters 
            1,  # At which granularity to update loading bar
            "Chain for n = $(posterior.sample_size) (joint)....")  # Loading bar. Minimum update interval: 1 second
    end 

    pointers = mcmc.pointers
    S_curr, S_prop = intialise_states!(pointers, S_init)
    γ_curr = γ_init 

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 
    γ_acc_count = 0
    reset_counts!(mcmc.move) # Reset counts for mcmc move (tracks acceptances)

    # Initialise aux data 
    initialise!(mcmc.aux, S_curr, γ_curr, posterior.S_prior, posterior.sample_size)
    
    # Initialise sufficient statistic 
    dist, data = (posterior.dist, posterior.data)
    suff_stats = mcmc.suff_stats
    suff_stats[1] = sum(dist(S_curr, x) for x in data)

    # Storage for sample statistic 
    sample_suff_stat = zeros(length(sample_out_S))

    while sample_count ≤ length(sample_out_S)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out_S[sample_count] = deepcopy(S_curr)
            @inbounds sample_out_gamma[sample_count] = γ_curr 
            @inbounds sample_suff_stat[sample_count] = suff_stats[1]
            sample_count += 1
        end 
        # Accept-reject move
        accept_reject_mode!(
            S_curr, S_prop, 
            γ_curr, pointers, 
            mcmc, posterior
        )
        # Accept-reject dispersion 
        γ_curr, was_acc = accept_reject_gamma(
            γ_curr, S_curr, 
            mcmc, 
            posterior
        )
        # println(S_curr)
        γ_acc_count += was_acc
        # println(S_prop)
        if loading_bar
            next!(iter)
        end 
    end 
    mcmc.γ_counts[1:2] = [γ_acc_count, i]
    return_states!(S_curr, S_prop, pointers)
    return sample_suff_stat
end 

function draw_sample(
    mcmc::T,
    posterior::S;
    desired_samples::Int=mcmc.desired_samples,
    args...
    ) where {T<:IexMcmcSampler,S<:Union{SisPosterior,SimPosterior}}

    sample_out_S = InteractionSequenceSample{Int}(undef, desired_samples)
    sample_out_gamma = zeros(desired_samples)
    suff_stat_sample = draw_sample!(
        sample_out_S,
        sample_out_gamma, 
        mcmc, 
        posterior;
        args...
        )
    return sample_out_S, sample_out_gamma, suff_stat_sample
end 

function (mcmc::T where {T<:IexMcmcSampler})(
    posterior::S;
    args...
    ) where {S<:Union{SisPosterior,SimPosterior}}
    S_sample, γ_sample, suff_stat_trace = draw_sample(
        mcmc, posterior;
        args...
    )
    return PosteriorMcmcOutput(
        S_sample, γ_sample, 
        suff_stat_trace, 
        posterior
    )
end 
