using InteractionNetworkModels, Plots, Distributions, JLD2 
using StatsPlots, Plots.Measures, StatsBase

cd("Z:/simulations/")
readdir()
chains, times = load(
    "iex_sim_50_match_dist_multi.jld2",
    "chains", "times"
) 

plot([x.suff_stat_trace for x in chains])
plot([length.(x.S_sample) for x in chains])

plot(chains, E)

posterior = chains[1].posterior
data = posterior.data
d = posterior.dist 

Ê, ind = sample_frechet_mean(data, d)
sum(x->d(x,Ê), data)