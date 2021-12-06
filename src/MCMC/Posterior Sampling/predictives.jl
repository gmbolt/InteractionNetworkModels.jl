using StatsBase

export SingleMissingPredictive, PosteriorPredictive, DistanceToModePredictive
export MeanInnerDimensionPredictive, OuterDimensionPredictive
export pred_missing, get_prediction, get_truth, was_correct, get_pred_accuracy

# ===========================
# Predicting missing entries 
# ===========================

struct SingleMissingPredictive
    S::InteractionSequence{Int}
    ind::Tuple{Int,Int}
    p::Vector{Float64}
end 

function Base.show(io::IO, pred::SingleMissingPredictive)
    title = "Missing Entry Predictive Distribution"
    println(io, title)
    println(io, "-"^length(title))
    println(io, "Observation: $(pred.S)")
    println(io, "Missing entry: $(pred.ind)")
end 

function pred_missing(
    S::InteractionSequence{Int},
    ind::Tuple{Int,Int},
    mcmc_post_out::SisPosteriorMcmcOutput{Int}
    )

    posterior = mcmc_post_out.posterior
    d = posterior.dist
    Sₓ = deepcopy(S)
    # Maps a mode to the V (num vertices) different distances to value S with different value in ind 
    dists_to_vals = Dict{InteractionSequence{Int},Vector{Real}}()
    V = length(posterior.V)
    # @show V
    dist_vec = zeros(V)
    μ_tmp = zeros(V)
    μ = zeros(V)

    for (mode, γ) in zip(mcmc_post_out.S_sample, mcmc_post_out.γ_sample)
        if mode ∉ keys(dists_to_vals)
            for x in 1:V
                Sₓ[ind[1]][ind[2]] = x
                dist_vec[x] = d(Sₓ, mode)
            end 
            dists_to_vals[mode] = dist_vec
        end 
        map!(x -> exp(- γ * x ), μ_tmp, dists_to_vals[mode])
        μ_tmp /= sum(μ_tmp)
        μ += μ_tmp
    end 

    μ /= sum(μ)

    return SingleMissingPredictive(S, ind, μ)
end

function pred_missing(
    S::InteractionSequence{Int},
    ind::Tuple{Int,Int},
    model::SIS{Int}
    )

    d, γ = (model.dist, model.γ)
    μ = zeros(length(model.V))
    Sₓ = deepcopy(S)
    i,j = ind
    for x in model.V
        Sₓ[i][j] = x 
        μ[x] = exp(-γ * d(Sₓ, model.mode))
    end 
    μ /= sum(μ)
    return SingleMissingPredictive(S, ind, μ)
end 

function get_prediction(
    predictive::SingleMissingPredictive
    )
    max_prob = maximum(predictive.p)  # MAP 
    vals = findall(predictive.p .== max_prob) # Vertices with max MAP
    pred = rand(vals) # Choose randomly from said vertices
    H = entropy(predictive.p) # Evaluate entropy 
    return pred, H
end 

function get_truth(
    predictive::SingleMissingPredictive
    )   
    i,j = predictive.ind
    return predictive.S[i][j]
end 

was_correct(predictive::SingleMissingPredictive) = (get_prediction(predictive)[1] == get_truth(predictive))

function get_pred_accuracy(
    predictives::Vector{SingleMissingPredictive}
    )
    return sum(was_correct.(predictives))/length(predictives)
end 

# ===============================
# Summary statistics predictives 
# ===============================

struct PosteriorPredictive
    posterior::SisPosteriorMcmcOutput
end 

function draw_sample!(
    out::Vector{InteractionSequence{Int}},
    mcmc::SisMcmcSampler,
    predictive::PosteriorPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(view(out,i:i), mcmc, model)
    end 
end 

function draw_sample!(
    out::Vector{InteractionSequenceSample{Int}},
    mcmc::SisMcmcSampler,
    predictive::PosteriorPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(out[i], mcmc, model)
    end 
end 


function draw_sample(
    mcmc::SisMcmcSampler, 
    predictive::PosteriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
    )

    if n_reps == 1 
        out = InteractionSequenceSample{Int}(undef, n_samples)
    else 
        out = [InteractionSequenceSample{Int}(undef, n_reps) for i in 1:n_samples]
    end 
    
    draw_sample!(out, mcmc, predictive)

    return out 
end 

struct DistanceToModePredictive 
    posterior::SisPosteriorMcmcOutput
end 


function draw_sample!(
    out::Vector{Float64},
    mcmc::SisMcmcSampler,
    predictive::DistanceToModePredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = d(sample_store[1], model.mode)
    end 
end 

function draw_sample!(
    out::Vector{Vector{Float64}},
    mcmc::SisMcmcSampler,
    predictive::DistanceToModePredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> d(x, model.mode), sample_store)
    end 
end 

function draw_sample(
    mcmc::SisMcmcSampler, 
    predictive::DistanceToModePredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
    )

    if n_reps == 1 
        out = zeros(n_samples)
    else 
        out = [zeros(n_reps) for i in 1:n_samples]
    end 
    
    draw_sample!(out, mcmc, predictive)

    return out 
end 

struct MeanInnerDimensionPredictive
    posterior::SisPosteriorMcmcOutput
end 

struct OuterDimensionPredictive 
    posterior::SisPosteriorMcmcOutput
end 

function draw_sample!(
    out::Vector{Int},
    mcmc::SisMcmcSampler,
    predictive::MeanInnerDimensionPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = mean(lenth.(sample_store[1]))
    end 
end 

function draw_sample!(
    out::Vector{Int},
    mcmc::SisMcmcSampler,
    predictive::OuterDimensionPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]

    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = length(sample_store[1])
    end 
end 

function draw_sample!(
    out::Vector{Vector{Int}},
    mcmc::SisMcmcSampler,
    predictive::MeanInnerDimensionPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> mean(length.(x)), sample_store)
    end 
end 

function draw_sample!(
    out::Vector{Vector{Int}},
    mcmc::SisMcmcSampler,
    predictive::OuterDimensionPredictive
    )

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample 
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = SIS(S_sample[ind],γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> length(x), sample_store)
    end 
end 

function draw_sample(
    mcmc::SisMcmcSampler, 
    predictive::Union{MeanInnerDimensionPredictive,OuterDimensionPredictive};
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
    )

    if n_reps == 1 
        out = zeros(Int, n_samples)
    else 
        out = [zeros(Int, n_reps) for i in 1:n_samples]
    end 
    
    draw_sample!(out, mcmc, predictive)

    return out 
end 