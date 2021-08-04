using Distributions, InvertedIndices, StatsBase, ProgressMeter

function MyPkg.iex_mcmc_within_gibbs_update!(
    posterior::SimPosterior{T},
    S_curr::Vector{Path{T}}, 
    γ_curr::Float64,
    i::Int, 
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    mcmc_sampler::SimMcmcSampler, # Sampler for auxiliary variables 
    aux_data::InteractionSequenceSample{T}
    ) where {T<:Union{Int, String}}

    S_prop = deepcopy(S_curr)

    @inbounds n = length(S_curr[i])

    δ = rand(1:ν)
    d = rand(0:min(n-1,δ))

    if T==Int
        I_prop, log_ratio = imcmc_prop_sample_edit_informed(S_curr[i], δ, d, P)
    else 
        I_prop, log_ratio = imcmc_prop_sample_edit_informed(S_curr[i], δ, d, P, vertex_map, vertex_map_inv)
    end 

    @inbounds S_prop[i] = I_prop

    # @show S_curr, S_prop

    # Sample auxiliary data 
    aux_model = SIM(S_prop, γ_curr, posterior.dist, posterior.V, K_inner=posterior.K_inner, K_outer=posterior.K_outer)
    draw_sample!(aux_data, mcmc_sampler, aux_model)
    
    log_lik_ratio = - γ_curr * (
            sum_of_dists(posterior.data, S_prop, posterior.dist)
            -sum_of_dists(posterior.data, S_curr, posterior.dist)
        )

    # @show aux_data
    aux_log_lik_ratio = -γ_curr * (
            sum_of_dists(aux_data, S_curr, posterior.dist) - sum_of_dists(aux_data, S_prop, posterior.dist)
        )
    # @show log_lik_ratio, aux_log_lik_ratio

    # EXTRA MULTINOMIAL TERM 
    log_multinom_ratio = log_multinomial_ratio(S_curr, S_prop)

    log_α = (
        posterior.S_prior.γ *(
            sum_of_dists(posterior.S_prior, S_curr)
            - sum_of_dists(posterior.S_prior, S_prop)
        )
        + log_lik_ratio + aux_log_lik_ratio + log_ratio + log_multinom_ratio
    ) 
    # println("Gibbs:")
    if log(rand()) < log_α
        @inbounds S_curr[i] = copy(S_prop[i])
        # println("$(i) value proposal $(tmp_prop) was accepted")
        return 1
        
    else
        # println("$(i) value proposal $(tmp_prop) was rejected")
        return 0
    end 
end 

function iex_mcmc_within_gibbs_scan!(
    posterior::SimPosterior{T},
    S_curr::Vector{Path{T}}, 
    γ_curr::Float64, 
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    mcmc_sampler::SimMcmcSampler, # Sampler for auxiliary variables 
    aux_data::InteractionSequenceSample{T}
    ) where {T<:Union{Int, String}}

    N = length(S_curr)
    count = 0
    for i in 1:N
        count += iex_mcmc_within_gibbs_update!(
            posterior, 
            S_curr,γ_curr, 
            i,
            ν,
            P, vertex_map, vertex_map_inv, 
            mcmc_sampler,
            aux_data)
    end 
    return count
end 

function MyPkg.iex_mcmc_mode(
    posterior::SimPosterior{T},
    mcmc_sampler::SimMcmcSampler,
    γ_fixed::Float64;
    S_init::Vector{Path{T}}=sample_frechet_mean(posterior.data, poseterior.dist),
    desired_samples::Int=100, # MCMC parameters...
    burn_in::Int=100,
    lag::Int=1,
    ν::Int=1, 
    path_dist::PathDistribution{T}=PathPseudoUniform(posterior.V, TrGeometric(0.6, 1, 10)),
    β = 0.0,
    α = 0.0
    ) where {T<:Union{Int, String}}


    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag

    iter = Progress(req_samples, 1, "Chain for γ = $(γ_fixed) and n = $(posterior.sample_size) (mode conditional)....")  # Loading bar. Minimum update interval: 1 second
    
    # Intialise 
    S_sample = Vector{Vector{Path{T}}}(undef, req_samples) # Storage of samples
    S_curr = S_init
    γ_curr = γ_fixed
    count = 0 
    acc_count = 0
    gibbs_scan_count = 0
    gibbs_tot_count = 0
    gibbs_acc_count = 0 
    aux_data = [[T[]] for i in 1:posterior.sample_size]

    # tmp_inds = Int[]
    # Vertex distribution for proposal 
    # μ = get_vertex_proposal_dist(posterior)
    # if T == String 
    #     vdist = StringCategorical(posterior.V, μ)
    # elseif T == Int
    #     vdist = Categorical(μ)
    # else 
    #     error("Path eltype not recognised for defining vertex proposal dist.")
    # end 

    # Pairwise occurence probability matrix (and vertex maps)
    P, vmap, vmap_inv = get_informed_proposal_matrix(posterior.data, α)


    # Bounds for uniform sampling of number of interactions
    lb(x::Vector{Path{T}}) = max( ceil(Int, length(x)/2), length(x) - ν_outer )
    ub(x::Vector{Path{T}}) = min(posterior.K_outer, 2*length(x), length(x) + ν_outer)

    probs_gibbs = 1/(2+1) + β
    
    for i in 1:req_samples
        
        if rand() < probs_gibbs
            gibbs_scan_count += 1
            gibbs_tot_count += length(S_curr)
            gibbs_acc_count += iex_mcmc_within_gibbs_scan!(
                posterior, # Target
                S_curr, γ_curr, # State values
                ν, 
                P, vmap, vmap_inv, 
                mcmc_sampler,
                aux_data) # Gibbs sampler parameters
            S_sample[i] = copy(S_curr)
            next!(iter)
        
        else 
            count +=  1

            if length(S_curr) == 1
                S_prop, log_ratio = imcmc_insert_prop_sample(
                    S_curr, path_dist
                )
                log_ratio *= 0.5
            elseif length(S_curr) == posterior.K_outer
                S_prop, log_ratio = imcmc_delete_prop_sample(
                    S_curr, path_dist
                )
                log_ratio *= 0.5
            elseif rand() < 0.5 # w.p. 1/2 insert a new interaction randomly
                S_prop, log_ratio = imcmc_insert_prop_sample(
                    S_curr,
                    path_dist
                )
            else # w.p. 1/2 delete an interaction randomly
                S_prop, log_ratio = imcmc_delete_prop_sample(
                    S_curr, 
                    path_dist
                )
            end 

            # Sample auxiliary data 
            aux_model = SIM(S_prop, γ_curr, posterior.dist, posterior.V, K_inner=posterior.K_inner, K_outer=posterior.K_outer)
            draw_sample!(aux_data, mcmc_sampler, aux_model)

            # Accept reject
            log_lik_ratio = - γ_curr * (
            sum_of_dists(posterior.data, S_prop, posterior.dist)
            -sum_of_dists(posterior.data, S_curr, posterior.dist)
                )
            
            aux_log_lik_ratio = -γ_curr * (
                    sum_of_dists(aux_data, S_curr, posterior.dist)
                    - sum_of_dists(aux_data, S_prop, posterior.dist)
                )

            # EXTRA MULTINOMIAL TERM 
            log_multinom_ratio = log_multinomial_ratio(S_curr, S_prop)

            log_α = (
                posterior.S_prior.γ *(
                    sum_of_dists(posterior.S_prior, S_curr)
                    - sum_of_dists(posterior.S_prior, S_prop)
                )
                + log_lik_ratio + aux_log_lik_ratio + log_ratio + log_multinom_ratio
            ) 

            # println("Transdim:")
            # @show log_lik_ratio, aux_log_lik_ratio

            if log(rand()) < log_α
                S_sample[i] = copy(S_prop)
                S_curr = copy(S_prop)
                acc_count += 1
            else 
                S_sample[i] = copy(S_curr)
            end 
            next!(iter)
        end 

    end 
    p_measures = Dict(
        "Proportion Gibbs moves" => gibbs_scan_count/(count + gibbs_scan_count),
        "Trans-dimensional move acceptance probability" => acc_count/count,
        "Gibbs move acceptance probability" => gibbs_acc_count/gibbs_tot_count
    )
    output = SimPosteriorModeConditionalMcmcOutput(
        γ_fixed, 
        S_sample[(burn_in+1):lag:end],
        posterior.dist,
        posterior.S_prior, 
        posterior.data,
        p_measures
    )
    return output
end 
