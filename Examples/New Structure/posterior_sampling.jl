using InteractionNetworkModels, StructuredDistances, Distributions
using Plots

E = [[1,2,1,2],
    [1,2,1],
    [3,4,3], 
    [3,4], 
    [1,2], 
    [1,2,1],
    [1,2,3],
]
d = MatchingDistance(FastLCS(101))
K_inner, K_outer = (DimensionRange(1,10), DimensionRange(1,25))
model = SIM(
    E, 3.5, 
    d,
    1:10,
    K_inner, K_outer
)

# Auxiliary sampler 
β = 0.7
mcmc_move = InvMcmcMixtureMove(
    (
        EditAllocationMove(ν=1),
        InsertDeleteMove(ν=1, len_dist=TrGeometric(0.8, 1, model.K_inner.u))
    ),
    (β, 1-β)
)
mcmc_sampler = InvMcmcSampler(
    mcmc_move,
    lag=30, burn_in=3000
    )

@time draw_sample(
    mcmc_sampler,
    model, desired_samples=5000, burn_in=0, lag=1
)

@time data = mcmc_sampler(
    model,
    desired_samples=50,
    lag=500,
    burn_in=10000
)

E_prior = SIM(E, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,7.0)

posterior = SimPosterior(data, E_prior, γ_prior)

# Construct posterior sampler
β = 0.8
mode_move = InvMcmcMixtureMove(
    (
        EditAllocationMove(ν=1),
        InsertDeleteMove(ν=1, len_dist=TrGeometric(0.8, 1, model.K_inner.u))
    ),
    (β, 1-β)
)

posterior_sampler = IexMcmcSampler(
    mode_move, mcmc_sampler,
    ε=0.3
)

x = posterior_sampler(posterior, desired_samples=100)
x
plot(x, E)
acceptance_prob(posterior_sampler)