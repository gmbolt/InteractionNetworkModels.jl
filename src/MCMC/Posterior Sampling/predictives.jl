using StatsBase

export SingleMissingPredictive
export pred_missing, get_prediction, get_truth

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

    m = length(mcmc_post_out.S_sample)
    μ ./= m

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