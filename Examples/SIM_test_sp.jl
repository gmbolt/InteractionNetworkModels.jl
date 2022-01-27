using InteractionNetworkModels, Distributions, BenchmarkTools, Plots

# The Model(s)
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
V = 1:20

d_lsp = MatchingDist(FastLSP(100))
# d_f = FastMatchingDist(FastLCS(100), 51)

K_inner = DimensionRange(2, 50)
K_outer = DimensionRange(1, 50)

model = SIM(S, 3.2, d_lsp, V, K_inner, K_outer)

# model_f = SIM(S, 4.0, d_f, V, 50, 50)

mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=5, β=0.6, ν_td=3,  
    len_dist=TrGeometric(0.1, model.K_inner.l, model.K_inner.u),
    lag=1,
    K=200
)
mcmc_sampler_sp = SimMcmcInsertDeleteSubpath(
    ν_ed=4, β=0.6, ν_td=3,  
    len_dist=TrGeometric(0.1, model.K_inner.l, model.K_inner.u),
    lag=1,
    K=200
)

@time out=mcmc_sampler(
    model, 
    lag=1, 
    init=model.mode, 
    desired_samples=1000,
    burn_in=0
)
plot(out)


@time out_sp=mcmc_sampler_sp(
    model, 
    lag=1, 
    init=model.mode, 
    desired_samples=1000,
    burn_in=0
)
plot!(out_sp)

summaryplot(out)
summaryplot!(out_sp)
