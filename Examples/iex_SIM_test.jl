using Distributed 
@everywhere begin 
    using Pkg
    # Pkg.activate("C:/users/boltg/.julia/dev/")
    Pkg.activate("/home/boltg/.julia/dev/")
end
using Distributed, JLD2, StatsBase, Distances
@everywhere using InteractionNetworkModels, Distributions

# Script inputs 
if (length(ARGS)==0)
    n_samples=10
    n_reps = nworkers()
    n_train = 20
elseif (length(ARGS)==1)
    n_samples = parse(Int, ARGS[1])
    n_reps = nworkers()
    n_train = 20
elseif (length(ARGS)==2)
    n_train = parse(Int, ARGS[1])
    n_samples = parse(Int, ARGS[2])
    n_reps = nworkers()
else
    n_train = parse(Int, ARGS[1])
    n_samples = parse(Int, ARGS[2])
    n_reps = parse(Int, ARGS[3])
end 

println("Making model...")

E = [[1,2,1,2],
    [1,2,1],
    [3,4,3], 
    [3,4], 
    [1,2], 
    [1,2,1],
    [1,2,3],
    [4,5],
    [7,8]
]
γ = 2.6
V = 1:10
d = MatchingDist(FastLCS(101))
K_inner, K_outer = (DimensionRange(2,100), DimensionRange(1,25))

model = SIM(
    E, γ, 
    d,
    V,
    K_inner, K_outer
)

mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=6, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.8, 1, model.K_inner.u),
    burn_in=5000, lag=50
)

println("Sampling data...")
@time mcmc_out = mcmc_sampler(
    model,
    desired_samples=50,
    lag=500,
    burn_in=10000
)

println("Constructing posterior...")
data = mcmc_out.sample
E_prior = SIM(E, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,7.0)
posterior = SimPosterior(data, E_prior, γ_prior)


# Construct posterior sampler
posterior_sampler = SimIexInsertDelete(
    mcmc_sampler,
    ν_ed=1, ν_td=1, len_dist=TrGeometric(0.8, K_inner.l, K_inner.u),
    ε=0.1,
    K=200,
    α=0.0,
    desired_samples=n_samples, burn_in=0, lag=1
)

println("Setting-up mapper to workers....")

@everywhere function f(
    posterior::SimPosterior,
    mcmc_sampler::SimPosteriorSampler,
    S_init::InteractionSequence,
    γ_init::Float64
    )

    # Small run for precompile 
    @time mcmc_sampler(
        posterior,
        S_init=S_init,
        γ_init=γ_init,
        desired_samples=5,
        burn_in=0, lag=1,
        loading_bar=false,
        aux_init_at_prev=true
    )
    out = @timed mcmc_sampler(
        posterior, 
        S_init=S_init,
        γ_init=γ_init,
        loading_bar=true,
        aux_init_at_prev=true
    )
    return (chain=out[1], time=out[2])
end 

Ê = sample_frechet_mean(posterior.data, posterior.dist)[1]
pars = [(
    post=posterior, 
    post_sampler=posterior_sampler,
    S_init=Ê,
    γ_init=γ
    ) for i in 1:n_reps
]

println("Sampling....")
out = pmap(x->f(x.post, x.post_sampler, x.S_init, x.γ_init), pars)

chains = [x.chain for x in out]
times = [x.time/60/60 for x in out]
# cd("Z:/simulations/")
cd("/luna/simulations")
save(
    "iex_sim_$(n_train)_match_dist_multi.jld2",
    "chains", chains,
    "times", times 
)