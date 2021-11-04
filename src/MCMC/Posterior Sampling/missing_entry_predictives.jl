export pred_missing

function pred_missing(
    S::InteractionSequence{Int},
    ind::Tuple{Int,Int},
    mcmc_post_out::SisPosteriorMcmcOutput{Int}
    )

    posterior = mcmc_post_out.posterior
    d = posterior.dist
    Sₓ = deepcopy(S)
    # Maps a mode to the V (num vertices) different distances to sample with different value in ind 
    dists_to_sample = Dict{InteractionSequence{Int},Vector{Real}}()
    V = length(posterior.V)
    # @show V
    dist_vec = zeros(V)
    μ_tmp = zeros(V)
    μ = zeros(V)

    for (mode, γ) in zip(mcmc_post_out.S_sample, mcmc_post_out.γ_sample)
        if mode ∉ keys(dists_to_sample)
            for x in 1:V
                Sₓ[ind[1]][ind[2]] = x
                dist_vec[x] = d(Sₓ, mode)
            end 
            dists_to_sample[mode] = dist_vec
        end 
        map!(x -> exp(- γ * x ), μ_tmp, dists_to_sample[mode])
        μ_tmp /= sum(μ_tmp)
        μ += μ_tmp
    end 

    m = length(mcmc_post_out.S_sample)
    μ ./= m

    return μ

end