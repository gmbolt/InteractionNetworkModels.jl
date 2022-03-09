using InteractionNetworkModels, BenchmarkTools, Plots, Distributions, StatsPlots
using Distances, Distances

d = Hamming()

mode = rand(0:1,10,10)
model = CER(mode, 0.1)

mode = rand(Bool,10,10)
model = CER(mode, 0.1)

out = draw_sample(model, 100)

plot(map(x->d(x,mode), out))

G_prior = CER(mode, 0.5)
α_prior = Beta(1,10)
plot(α_prior)

mode = draw_sample(G_prior, 1)[1]
α = rand(α_prior)
model = CER(mode, α)
data = draw_sample(model, 100)

data = [convert.(Int,x) for x in data]
posterior = CerPosterior(data, G_prior, α_prior)

