using Distributions, InvertedIndices, StatsBase, ProgressMeter

export iex_mcmc_mode, iex_mcmc_gamma
export get_cooccurrence_matrix, get_cooccurrence_prob_matrix, plot_cooccurrence_graph
export merge_move_with_kick_sampler, separation_move_with_kick_sampler
export iex_mcmc_within_gibbs_update!, iex_mcmc_within_gibbs_scan!
export rand_multivariate_bernoulli
export get_informed_proposal_matrix


# Functions for insertions/deletions
# ----------------------------------

function imcmc_insert_prop_sample(
    S_curr, path_dist
    )

    i = rand(1:(length(S_curr)+1))
    S_prop = copy(S_curr)
    I′ = rand(path_dist)
    insert!(S_prop, i, I′)
    log_ratio = - logpdf(path_dist, I′)

    return S_prop, log_ratio
end 

function imcmc_delete_prop_sample(
    S_curr, path_dist
    )

    i = rand(1:length(S_curr))
    S_prop = copy(S_curr)
    deleteat!(S_prop, i)
    log_ratio = logpdf(path_dist, S_curr[i])

    return S_prop, log_ratio
end 

# const print_notes = false


# Informed iMCMC path proposal. We add a specialised proposal to be used when updating 
# interactions during the Gibbs scan. We do so by extending the imcmc_prop_sample_edit_informed()
# function, making use of multiple dispatch. 

# NOTE μ_cusum must have 0.0 first entry, and for n bins μ_cusum must be of size n+1
function rand_multivariate_bernoulli(μ_cusum::Vector{Float64})
    @assert μ_cusum[1] == 0 "First entry must be 0.0 (for differencing to find probabilities)."
    β = rand()
    for i in 1:length(μ_cusum)
        if β < μ_cusum[i]
            return i-1, μ_cusum[i]-μ_cusum[i-1]
        else 
            continue 
        end 
    end 
end 

# NOTE!!! - We have separate functions for when 
# (i) the data type for interactions are Int 
# (ii) data types for interactions are String
# 
# In case (i) we need not worry about mapping across to vertices 
# In case (ii) we must map integers to vertices 


function InteractionNetworkModels.imcmc_prop_sample_edit_informed(
    I_curr::Path{Int}, 
    δ::Int,
    d::Int, 
    P::CumCondProbMatrix
    ) 

    n = length(I_curr)
    m = n + δ - 2*d 

    log_ratio_entries = 0.0

    ind_del = StatsBase.seqsample_a!(1:n, zeros(Int, d))
    ind_add = StatsBase.seqsample_a!(1:m, zeros(Int, δ - d))

    I_prop = Vector{Int}(undef, m)

    # Set up distribution to sample new vertices (via cooccurrence)
    V = size(P)[2]
    ind_vertices = [i ∈ I_curr[Not(ind_del)] for i in 1:V] 
    μ_cusum = sum(P[:, ind_vertices], dims=2)[:] ./ sum(ind_vertices)
    # @show μ_cusum, ind_del, ind_add

    # Additions
    for i ∈ ind_add
        I_prop[i], prob_tmp = rand_multivariate_bernoulli(μ_cusum)
        log_ratio_entries += -log(prob_tmp)
    end 

    # Deletions
    I_prop[Not(ind_add)] = I_curr[Not(ind_del)]
    for i ∈ ind_del
        val = I_curr[i] # Value which was deleted
        log_ratio_entries += log(μ_cusum[val+1] - μ_cusum[val]) # Add this to the probability
    end 

    log_ratio = log(min(n-1, δ) + 1) - log(min(m-1, δ) + 1) + log_ratio_entries

    return I_prop, log_ratio

end 

function InteractionNetworkModels.imcmc_prop_sample_edit_informed(
    I_curr::Path{String}, 
    δ::Int,
    d::Int, 
    P::CumCondProbMatrix,
    vertex_map::Dict{Int, String},
    vertex_map_inv::Dict{String, Int}
    ) 

    n = length(I_curr)
    m = n + δ - 2*d 

    log_ratio_entries = 0.0

    ind_del = StatsBase.seqsample_a!(1:n, zeros(Int, d))
    ind_add = StatsBase.seqsample_a!(1:m, zeros(Int, δ - d))

    I_prop = Vector{String}(undef, m)

    # Set up distribution to sample new vertices (via cooccurrence). This essentiall sums over the 
    # columns (equivalently rows) corresponding to the vertices being preserved. 
    ind_vertices = [vertex_map[i] ∈ I_curr[Not(ind_del)] for i in 1:length(vertex_map)] 
    μ_cusum = sum(P[:, ind_vertices], dims=2)[:] ./ sum(ind_vertices)
    # @show μ_cusum, ind_del, ind_add

    # Additions (and add (-)log probs)
    for i ∈ ind_add
        vertex_ind, prob_tmp = rand_multivariate_bernoulli(μ_cusum)
        I_prop[i] = vertex_map[vertex_ind]
        log_ratio_entries += -log(prob_tmp)
    end 

    # Deletions (and add log probs)
    I_prop[Not(ind_add)] = I_curr[Not(ind_del)] 
    # Finding log prob
    for i ∈ ind_del
        val = I_curr[i] # Value which was deleted
        vertex_ind = vertex_map_inv[val]  # Find index of value 
        log_ratio_entries += log(μ_cusum[vertex_ind+1] - μ_cusum[vertex_ind]) # Add this to the probability
    end 


    log_ratio = log(min(n-1, δ) + 1) - log(min(m-1, δ) + 1) + log_ratio_entries

    return I_prop, log_ratio

end 

function iex_mcmc_within_gibbs_update!(
    posterior::SisPosterior{T},
    S_curr::Vector{Path{T}}, 
    γ_curr::Float64,
    i::Int, 
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    mcmc_sampler::SisMcmcSampler, # Sampler for auxiliary variables 
    aux_data::InteractionSequenceSample{T},
    all_aux_data::InteractionSequenceSample{T}
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
    aux_model = SIS(S_prop, γ_curr, posterior.dist, posterior.V, posterior.K_inner, posterior.K_outer)
    # println("Initialising at:", aux_data[end])
    # aux_data = draw_sample(mcmc_sampler, aux_model, init=deepcopy(aux_data[end])) # Initialise at last val of previous
    # draw_sample!(aux_data, mcmc_sampler, aux_model, init=deepcopy(aux_data[end]))
    draw_sample!(aux_data, mcmc_sampler, aux_model)
    # push!(all_aux_data, deepcopy(aux_data)...)
    # println("\n")
    # for x in aux_data
    #     println(x)
    # end 
    # println("\n")
    
    log_lik_ratio = - γ_curr * (
            sum_of_dists(posterior.data, S_prop, posterior.dist)
            -sum_of_dists(posterior.data, S_curr, posterior.dist)
        )
    # @show aux_data
    aux_log_lik_ratio = -γ_curr * (
            sum_of_dists(aux_data, S_curr, posterior.dist) - sum_of_dists(aux_data, S_prop, posterior.dist)
        )
    # @show log_lik_ratio, aux_log_lik_ratio
    log_α = (
        posterior.S_prior.γ *(
            sum_of_dists(posterior.S_prior, S_curr)
            - sum_of_dists(posterior.S_prior, S_prop)
        )
        + log_lik_ratio + aux_log_lik_ratio + log_ratio
    ) 
    # @show log_lik_ratio, aux_log_lik_ratio, log_ratio
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
    posterior::SisPosterior{T},
    S_curr::Vector{Path{T}}, 
    γ_curr::Float64, 
    ν::Int,
    P::CumCondProbMatrix,
    vertex_map::Dict{Int,T},
    vertex_map_inv::Dict{T,Int},
    mcmc_sampler::SisMcmcSampler, # Sampler for auxiliary variables 
    aux_data::InteractionSequenceSample{T},
    all_aux_data::InteractionSequenceSample{T}
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
            aux_data, 
            all_aux_data)
    end 
    return count
end 


# function get_vertex_proposal_dist(
#     posterior::SisPosterior{T}
#     ) where {T<:Union{Int,String}}

#     x = vec(posterior.data)
#     c = countmap(x)
#     μ = [v in keys(c) ? c[v] : 0 for v in posterior.V] 
#     μ /= sum(μ)
#     return μ

# end 


function get_cooccurrence_matrix(data::Vector{Vector{Path{Int}}})
    V = maximum(unique(vec(data)))
    C = zeros(Int, V, V)
    for S in data 
        C_tmp = zeros(Int, V, V)
        for I in S
            tmp_counts = counts(I, 1:V)
            vals = unique(I)
            # @show vals
            for j in 1:length(vals)
                for i in 1:(j-1)
                    C_tmp[vals[i],vals[j]] += 1 
                    C_tmp[vals[j],vals[i]] += 1 
                end 
                if tmp_counts[vals[j]] > 1 # If vertex appears more than once add to it's diag
                    C_tmp[vals[j], vals[j]] += 1
                end 
            end
        end 
        C += Int.(C_tmp .> 0)
    end 
    d = Dict(i => i for i in 1:V)
    return C, d, d
end 

function get_cooccurrence_matrix(data::Vector{Vector{Path{String}}})
    vertex_set = unique(vec(data)) 
    V = length(vertex_set)
    C = zeros(Int, V, V)
    ind_map = Dict{String, Int}(v => i for (i,v) in enumerate(vertex_set))
    for S in data 
        C_tmp = zeros(Int, V, V)
        for I in S
            tmp_counts = countmap(I)
            vals = unique(I)
            for j in 1:length(vals)
                for i in 1:(j-1)
                    C_tmp[ind_map[vals[i]],ind_map[vals[j]]] += 1 # only filling upper tri
                    C_tmp[ind_map[vals[j]],ind_map[vals[i]]] += 1 # only filling upper tri
                end 
                if tmp_counts[vals[j]] > 1 # If vertex appears more than once add to it's diag
                    C_tmp[ind_map[vals[j]], ind_map[vals[j]]] += 1
                end 
            end
        end 
        C += Int.(C_tmp .> 0)
    end 
    ind_map_inv = Dict(val => key for (val, key) in zip(values(ind_map), keys(ind_map)))
    return C, ind_map, ind_map_inv
end

function get_cooccurrence_prob_matrix(C::Matrix{Int}, α::Real)
    
    @assert issymmetric(C) "Input matrix should be symmetric."

    V = size(C)[1]
    P = convert(Array{Float64,2}, C)

    for i in 1:V
        Z = sum(P[:,i])
        if  Z == 0.0
            P[:,i] = fill(1/V, V)
        else 
            P[:,i] = ((P[:,i] ./ Z) .+ α) ./ (1 + V*α)
        end 
    end 
    return CondProbMatrix(P)
end 

function get_informed_proposal_matrix(data::Vector{Vector{Path{Int}}}, α::Real)
    C, vmap, vmap_inv = get_cooccurrence_matrix(data)
    P = get_cooccurrence_prob_matrix(C, α)
    return cumsum(P), vmap, vmap_inv
end 

function plot_cooccurrence_graph(data::Vector{Vector{Path{Int}}})
    C, vmap, vmap_inv = get_cooccurrence_matrix(data)
    C₀ = copy(C)
    C₀[diagind(C₀)] .= 0
    fig = graphplot(
        C₀, names=vmap, 
        edge_width= (s,d,w) -> 2*C[s,d]/maximum(C),
        markersize=0.15, shape=:circle,
        fontsize=10,
        msw=0
    )
    display(fig)
end 




function iex_mcmc_mode(
    posterior::SisPosterior{T},
    mcmc_sampler::SisMcmcSampler,
    γ_fixed::Float64;
    S_init::Vector{Path{T}}=sample_frechet_mean(posterior.data, posterior.dist),
    desired_samples::Int=100, # MCMC parameters...
    burn_in::Int=100,
    lag::Int=1,
    ν::Int=1, 
    path_dist::PathDistribution{T}=PathPseudoUniform(posterior.V, TrGeometric(0.7, 1, posterior.K_inner)),
    β = 0.6, # Probability of Gibbs move
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
    all_aux_data = InteractionSequence{T}[]
    # Initialise the aux_data with a large burn_in 
    aux_model = SIS(S_curr, γ_curr, posterior.dist, posterior.V, posterior.K_inner, posterior.K_outer)
    draw_sample!(aux_data, mcmc_sampler, aux_model, burn_in=1000)
    
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
    
    for i in 1:req_samples
        
        if rand() < β
            gibbs_scan_count += 1
            gibbs_tot_count += length(S_curr)
            gibbs_acc_count += iex_mcmc_within_gibbs_scan!(
                posterior, # Target
                S_curr, γ_curr, # State values
                ν, 
                P, vmap, vmap_inv, 
                mcmc_sampler,
                aux_data, 
                all_aux_data) # Gibbs sampler parameters
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
            aux_model = SIS(S_prop, γ_curr, posterior.dist, posterior.V, posterior.K_inner, posterior.K_outer)
            # println("Initialising at:", aux_data[end])
            # aux_data = draw_sample(mcmc_sampler, aux_model, init=deepcopy(aux_data[end])) # Initialise at last val of previous
            draw_sample!(aux_data, mcmc_sampler, aux_model)
            # push!(all_aux_data, deepcopy(aux_data)...)
            # Accept reject
            log_lik_ratio = - γ_curr * (
            sum_of_dists(posterior.data, S_prop, posterior.dist)
            -sum_of_dists(posterior.data, S_curr, posterior.dist)
                )
            
            aux_log_lik_ratio = -γ_curr * (
                    sum_of_dists(aux_data, S_curr, posterior.dist)
                    - sum_of_dists(aux_data, S_prop, posterior.dist)
                )

            log_α = (
                posterior.S_prior.γ *(
                    sum_of_dists(posterior.S_prior, S_curr)
                    - sum_of_dists(posterior.S_prior, S_prop)
                )
                + log_lik_ratio + aux_log_lik_ratio + log_ratio
            ) 
            # println("Transdim:")
            # @show log_lik_ratio, aux_log_lik_ratio
            # @show log_lik_ratio, aux_log_lik_ratio, log_ratio
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
    output = SisPosteriorModeConditionalMcmcOutput(
        γ_fixed, 
        S_sample[(burn_in+1):lag:end],
        posterior.dist,
        posterior.S_prior, 
        posterior.data,
        p_measures
    )
    return output
end 



function iex_mcmc_gamma(
    posterior::SisPosterior{T},
    mcmc_sampler::SisMcmcSampler, 
    S_fixed::Vector{Path{T}};
    γ_init::Float64=3.5,
    desired_samples::Int=100, 
    burn_in::Int=0,
    lag::Int=1, 
    ε::Float64=0.1
    ) where {T<:Union{Int,String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag

    iter = Progress(req_samples, 1, "Chain for mode fixed and n = $(posterior.sample_size) (dispersion conditional)....")  # Loading bar. Minimum update interval: 1 second
    
    # Intialise 
    γ_sample = Vector{Float64}(undef, req_samples) # Storage of samples
    S_curr = S_fixed    
    γ_curr = γ_init 
    aux_data = [[T[]] for i in 1:posterior.sample_size]
    count = 0

    suff_stat = sum_of_dists(posterior.data, S_curr, posterior.dist) # Sum of distance to mode, fixed throughout

    for i in 1:req_samples
        # Sample proposed γ

        γ_prop = rand_reflect(γ_curr, ε, 0.0, Inf)

        # Generate auxiliary data (centered on proposal)
        aux_model = SIS(
            S_curr, γ_prop, 
            posterior.dist, 
            posterior.V, 
            posterior.K_inner, posterior.K_outer
            )
        draw_sample!(aux_data, mcmc_sampler, aux_model)

        # Accept reject

        log_lik_ratio = (γ_curr - γ_prop) * suff_stat
        aux_log_lik_ratio = (γ_prop - γ_curr) * sum_of_dists(aux_data, S_curr, posterior.dist)

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

    output = SisPosteriorDispersionConditionalMcmcOutput(
        S_fixed, 
        γ_sample[(burn_in+1):lag:end],
        posterior.γ_prior, 
        posterior.data,
        performance_measures
    )
    return output
end 
