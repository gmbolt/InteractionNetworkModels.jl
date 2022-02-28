using InteractionNetworkModels, Distributions, Plots, StatsPlots

mode = [1,1,1,1,1]
γ = 4.0
d = LCS()
V = 1:10
p = truncated(Poisson(10), 1, Inf)
p = TrGeometric(0.8, 1, 90)
model = DcSPF(mode, γ, d, p, V)

mcmc = DcSpfMcmcSampler(ν=1, lag=30)

@time sample = draw_sample(mcmc,model, desired_samples=1000)
sample

plot(map(x->d(x,mode), sample))

histogram(length.(sample))