using Distances, StructuredDistances, InteractionNetworkModels
using Plots, BenchmarkTools, Distributions, StatsPlots

d = Cityblock()

V = 9
τ = 0.2
mode = [rand() < 0.1 ? rand(1:4) : 0 for i in 1:V,j in 1:V]
γ = 1.9

model = SNF(mode, γ, d)
model = vectorise(model)

# Gibbs scan 
gibbs_move_rand = GibbsRandMove(ν=1)
gibbs_move = GibbsScanMove(ν=1)

mcmc = McmcSampler(gibbs_move_rand)
mcmc_scan = McmcSampler(gibbs_move)

@time x = mcmc(model, desired_samples=10000, lag=1)
plot(x)

sum(mode)

acceptance_prob(mcmc_scan)
acceptance_prob(mcmc)
typeof(model)

# Testing posterior sampler
@time out  = mcmc_scan(model, desired_samples=100, lag=5, burn_in=500)
plot(out)
similar(model, rand(1:10, 20), 1.0)

model.mode
data = out.sample 
γ_prior = Gamma(model.γ, 1)
plot(γ_prior)
G_prior = SNF(
    model.mode, 
    0.1, 
    model.d,
    directed=model.directed, 
    self_loops=model.self_loops
)
posterior = SnfPosterior(data, G_prior, γ_prior)
mode_move = GibbsScanMove(ν=1)
aux_mcmc = McmcSampler(GibbsRandMove(ν=1), burn_in=3000, lag=50)

@time x = aux_mcmc(model, desired_samples=100)
acceptance_prob(aux_mcmc)
plot(x)

mcmc_posterior = SnfPosteriorSampler(
    mode_move, aux_mcmc, posterior,
    ε=0.1, 
    aux_init_at_prev=true
)
mcmc_posterior.aux.data

@time x = mcmc_posterior(posterior, desired_samples=500, γ_init=4.0)

plot(x, model.mode)
x_sample_mat = get_sample_matrices(x)[end]

data

# Plot graph 
using CairoMakie, GraphMakie, Graphs

g = SimpleDiGraph(x_sample_mat[1])

f,ax,p = graphplot(
    g,
    node_color=:blue,
    edge_plottype=:linesegments,
    showaxis=false
)
hidedecorations!(ax)  # hides ticks, grid and lables
hidespines!(ax)  # hide the frame
f 

V = 15
g = SimpleDiGraph(
    [rand() < 0.5 ? true : false for i in 1:V, j in 1:V]
)

f, ax, p = graphplot(
    g,
    node_size=10,
    node_color=:red,
    edge_plottype=:linesegments
)
hidedecorations!(ax)  # hides ticks, grid and lables
hidespines!(ax)  # hide the frame
f