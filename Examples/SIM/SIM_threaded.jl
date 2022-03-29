using Pkg 
Pkg.activate("../../..")
using InteractionNetworkModels, Distributions, BenchmarkTools
using Distances, StructuredDistances
# The Model(s)

model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1,1,1,1], [2,2,2,2], [3,3,3,3,3,3]]
V = 1:20

d_lcs = MatchingDistance(FastLCS(100))
d_lcs_t = ThreadedMatchingDistance(LCS())

K_inner = DimensionRange(1, 10)
K_outer = DimensionRange(1, 50)


model = SIM(S, 5.0, d_lcs, V, K_inner, K_outer)
model_t = SIM(S, 5.0, d_lcs_t, V, K_inner, K_outer)

# model_f = SIM(S, 4.0, d_f, V, 50, 50)
mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=5, β=0.6, ν_td=3,  
    len_dist=TrGeometric(0.8, K_inner.l, K_inner.u),
    lag=1,
    K=200, 
    burn_in=1000
)

@time out=mcmc_sampler(
    model, 
    lag=20, 
    init=model.mode, 
    burn_in=0,
    desired_samples=200
)

@time out=mcmc_sampler(
    model_t, 
    lag=20, 
    init=model.mode, 
    burn_in=0,
    desired_samples=200
)