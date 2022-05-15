using InteractionNetworkModels, StructuredDistances, BenchmarkTools
using Plots
mode = [[1,2,1],[1,1,1]]
mode = [[1,2,1],[1]]
γ = 2.9
d_S = EditDistance(LCS())
d_E = MatchingDistance(LCS())
d_E = AfpMatchingDistance(FastLCS(100), 3.0)
V = 1:10

model = SIM(mode, γ, d_E, V, 10, 20)
β = 0.7
mcmc_move = InvMcmcMixtureMove(
    (
        EditAllocationMove(ν=1),
        InsertDeleteMove(ν=1, len_dist=TrGeometric(0.8, 1, model.K_inner.u))
    ),
    (β, 1-β)
)

mcmc = InvMcmcSampler(
    mcmc_move,
    burn_in=2000, lag=75
    )

x = mcmc(model, desired_samples=10000, lag=1, burn_in=0)
plot(x)
acceptance_prob(mcmc)
summaryplot(x)

x.sample[findall(length.(x.sample) .> 5)]

@btime draw_sample(mcmc, model, desired_samples=1000, burn_in=0, lag=1)

acceptance_prob(mcmc)


mcmc_old = SimMcmcInsertDelete(
    ν_ed=1, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.8, 1, model.K_inner.u),
    burn_in=2000, lag=75
)

@btime draw_sample(mcmc_old, model, desired_samples=1000, burn_in=0, lag=1)

x_old = mcmc_old(model, desired_samples=4000)
plot!(x_old)
summaryplot!(x_old)

x.sample
x_old.sample

x_old.sample

x.sample