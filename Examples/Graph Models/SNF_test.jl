using Distances, StructuredDistances, InteractionNetworkModels
using Plots, BenchmarkTools, Distributions

d = Cityblock()
V = 20
mode = rand(0:10, V, V)
γ = 5.4
model = SNF(mode, γ, d)

# Gibbs scan 
mcmc = SnfMcmcRandGibbs(ν=1)

@time out, a = draw_sample(mcmc, model, desired_samples=1000, lag=100, burn_in=0)
plot(map(x->d(x,model.mode), out))

out = mcmc(model)
plot(out)
@time x = mcmc(model, desired_samples=1000, lag=100)
plot(x)

