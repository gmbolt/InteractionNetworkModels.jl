using Distributions, InvertedIndices, StatsBase, ProgressMeter, RecipesBase

export iex_mcmc_mode, iex_mcmc_gamma, aux_term_eval_mode, rand_reflect

function rand_reflect(x, ε, l, u)
    ξ = ε * (2*rand() - 1)
    y = x + ξ
    if y < l 
        return 2*l - y
    elseif y > u 
        return 2*u - y
    else 
        return y 
    end 
end 

# Auxiliary sampler types 
function aux_term_eval_mode(
    mcmc::SpfInvolutiveMcmcEdit,
    I_curr::Path{T}, I_prop::Path{T},
    posterior::SpfPosterior{T},
    γ_curr::Float64,
    aux_model::SPF{T}
    ) where {T<:Union{Int,String}}

    @assert aux_model.mode == I_prop "Auxiliary model must have proposed value as mode."

    aux_log_lik_ratio = 0.0

    # Pointers for clear exposition
    aux_curr = mcmc.curr
    aux_prop = mcmc.prop
    ind_del = mcmc.ind_del
    ind_add = mcmc.ind_add
    vals = mcmc.vals
    ν = mcmc.ν

    # We are sampling from model with pars (I_prop, γ_curr)

    # Initialise at I_prop
    copy!(aux_curr, I_prop)
    copy!(aux_prop, aux_curr)
    # ind_del = zeros(Int, mcmc.ν)
    # ind_add = zeros(Int, mcmc.ν)
    # vals = rand(posterior.V, mcmc.ν)
    count = 0 

    # burn_in
    for i in 1:mcmc.burn_in
        was_accepted = accept_reject_edit!(
            aux_curr, aux_prop, 
            aux_model, 
            ν,
            ind_del, ind_add, vals
            )
        count+=was_accepted
    end 
    # For sample size, do lag-1 steps with no action, then accept reject and 
    # increment the auxiliary term 
    for i in 1:posterior.sample_size
        for j in 1:(mcmc.lag-1)
            was_accepted = accept_reject_edit!(
            aux_curr, aux_prop,
            aux_model,
            ν,
            ind_del, ind_add, vals
            )
        count+=was_accepted
        end 
        # Now value TO BE USED IN AUX TERM
        was_accepted = accept_reject_edit!(
            aux_curr, aux_prop, 
            aux_model, 
            ν,
            ind_del, ind_add, vals
            )
        count+=was_accepted
        
        aux_log_lik_ratio += -γ_curr * (
            posterior.dist(aux_curr, I_curr) 
            - posterior.dist(aux_curr, I_prop)
            )
    
    end 
    return aux_log_lik_ratio

end 

function iex_mcmc_mode(
    posterior::SpfPosterior{T}, 
    mcmc_sampler::SpfMcmcSampler,
    γ_fixed::Float64; 
    I_init::Path{T}=sample_frechet_mean(posterior.data, posterior.dist)[1],
    desired_samples::Int=100, 
    burn_in=0, 
    lag=1,
    ν=2
    )  where {T<:Union{Int, String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag

    iter = Progress(req_samples, 1, "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode marginal)....")  # Loading bar. Minimum update interval: 1 second
    
    # Intialise 
    I_sample = Vector{Path{T}}() # Storage of samples
    I_curr = copy(I_init)
    γ_curr = γ_fixed 
    # aux_data = Vector{Path{T}}(undef, length(posterior.data))
    I_count = 0
    aux_data = fill(T[], posterior.sample_size)
    aux_model = SPF(I_curr, γ_curr, posterior.dist, posterior.V, posterior.K)


    # Vertex distribution for proposal 
    μ = get_vertex_proposal_dist(posterior)
    if T == String 
        vdist = StringCategorical(posterior.V, μ)
    elseif T == Int
        vdist = Categorical(μ)
    else 
        error("Path eltype not recognised for defining vertex proposal dist.")
    end 

    for i in 1:req_samples
        # Sample proposed mode
        n = length(I_curr)  
        δ = rand(1:ν)
        d = rand(0:min(n-1,δ))
        
        I_prop, log_ratio = imcmc_prop_sample_edit_informed(I_curr, δ, d, vdist)  # Sample new mode

        # Generate auxiliary data (centered on proposal)
        aux_model = SPF(I_prop, γ_curr, posterior.dist, posterior.V, posterior.K)
        # aux_data = mcmc_sampler(aux_model, desired_samples=posterior.sample_size).sample
        sample!(mcmc_sampler, aux_model, aux_data)
        # Accept reject

        log_lik_ratio = -γ_curr * (
            sum_of_dists(posterior.data, I_prop, posterior.dist)
            - sum_of_dists(posterior.data, I_curr, posterior.dist)
        )
        aux_log_lik_ratio = -γ_curr * (
            sum_of_dists(aux_data, I_curr, posterior.dist)
            - sum_of_dists(aux_data, I_prop, posterior.dist)
        )
        # aux_log_lik_ratio = aux_term_eval_mode(
        #     mcmc_sampler,
        #     I_curr, I_prop, 
        #     posterior,
        #     γ_curr,
        #     aux_model
        # )
        log_α = (
            posterior.I_prior.γ * (
                sum_of_dists(posterior.I_prior, I_curr)
                - sum_of_dists(posterior.I_prior, I_prop)
            )
            + log_lik_ratio + aux_log_lik_ratio + log_ratio 
        )

        if log(rand()) < log_α
            copy!(I_curr, I_prop)
            I_count += 1
        else 
            copy!(I_prop, I_curr)
        end 
        push!(I_sample, copy(I_curr))
        next!(iter)

    end 
    performance_measures = Dict(
        "Mode acceptance probability" => I_count/req_samples
    )

    output = SpfPosteriorModeConditionalMcmcOutput(
        γ_fixed, 
        I_sample[(burn_in+1):lag:end],
        posterior.dist,
        posterior.I_prior, 
        posterior.data,
        performance_measures
    )
    return output
end 

function iex_mcmc_gamma(
    posterior::SpfPosterior{T}, 
    mcmc_sampler::SpfMcmcSampler,
    I_fixed::Path{T}; 
    γ_init::Float64=3.9,
    desired_samples::Int=100, 
    burn_in::Int=0, 
    lag::Int=1,
    ε::Float64=0.1
    )  where {T<:Union{Int, String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag

    iter = Progress(req_samples, 1, "Chain for mode = $(I_fixed) and n = $(posterior.sample_size) (dispersion marginal)....")  # Loading bar. Minimum update interval: 1 second
    
    # Intialise 
    γ_sample = Vector{Float64}(undef, req_samples) # Storage of samples
    I_curr = I_fixed    
    γ_curr = γ_init 
    aux_data = fill(T[], posterior.sample_size)
    count = 0

    suff_stat = sum_of_dists(posterior.data, I_curr, posterior.dist) # Sum of distance to mode, fixed throughout

    for i in 1:req_samples
        # Sample proposed γ

        γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

        # Generate auxiliary data (centered on proposal)
        aux_model = SPF(I_curr, γ_prop, posterior.dist, posterior.V, posterior.K)
        # aux_data = mcmc_sampler(a_model, desired_samples=posterior.sample_size).sample
        sample!(mcmc_sampler, aux_model, aux_data)
        # Accept reject

        log_lik_ratio = (γ_curr - γ_prop) * suff_stat
        aux_log_lik_ratio = (γ_prop - γ_curr) * sum_of_dists(aux_data, I_curr, posterior.dist)

        log_α = (
            logpdf(posterior.γ_prior, γ_prop) 
            - logpdf(posterior.γ_prior, γ_curr)
            + log_lik_ratio + aux_log_lik_ratio 
        )

        if log(rand()) < log_α
            γ_sample[i] = γ_prop
            γ_curr = γ_prop
            count += 1
        else 
            γ_sample[i] = γ_curr
        end 
        next!(iter)

    end 
    performance_measures = Dict(
        "Mode acceptance probability" => count/req_samples
    )

    output = SpfPosteriorDispersionConditionalMcmcOutput(
        I_fixed, 
        γ_sample[(burn_in+1):lag:end],
        posterior.γ_prior, 
        posterior.data,
        performance_measures
    )
    return output
end 

