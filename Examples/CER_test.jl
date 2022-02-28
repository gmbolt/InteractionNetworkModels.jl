using InteractionNetworkModels, BenchmarkTools, Plots, Distributions, StatsPlots
using Distances
mode = rand(0:1,10,10)
model = CER(mode, 0.1, true)

d = Binomial(20, 0.05)

plot(d)

SNF(ones(Int,10,10), 3.0, Hamming())
