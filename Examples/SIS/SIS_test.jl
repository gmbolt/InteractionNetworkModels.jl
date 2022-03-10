using InteractionNetworkModels, Distributions, BenchmarkTools, Distances, StructuredDistances
using Plots

# The Model(s)
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
V = 1:20
d = FastEditDistance(FastLCS(51),51)
model = SIS(S, 4.9, d, V, 50, 50)
# model = SIS(S, 2.9, d, V)

mcmc_sampler = SisMcmcInsertDelete(
    ν_ed=2, β=0.6, ν_td=2,  
    lag=1,
    K=200)

mcmc_sampler.prop_pointers

@time out = mcmc_sampler(
    model, 
    lag=20, 
    init=model.mode, 
    desired_samples=2000,
    burn_in=0
)

plot(out)
summaryplot(out)
S
out.sample

