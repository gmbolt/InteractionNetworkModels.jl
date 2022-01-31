using InteractionNetworkModels, Distributions, BenchmarkTools, Plots

# The Model(s)
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
V = 1:20

d_lcs = MatchingDist(FastLCS(100))
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
    K=200, 
    burn_in=1000
)
mcmc_sampler_sp = SimMcmcInsertDeleteSubpath(
    ν_ed=5, β=0.6, ν_td=3,  
    len_dist=TrGeometric(0.1, model.K_inner.l, model.K_inner.u),
    lag=1,
    K=200
)

@time out=mcmc_sampler(
    model, 
    lag=1, 
    init=model.mode, 
    desired_samples=1000
)
plot(out)
sample_frechet_var(out.sample, d_lcs, with_memory=true)

n = 1000
m = 4
samples = [draw_sample(mcmc_sampler, model, desired_samples=n) for i in 1:m]

mean_dists_summary(samples, d_lcs)


@time out_sp=mcmc_sampler_sp(
    model, 
    lag=1, 
    init=model.mode, 
    desired_samples=10000,
    burn_in=0
)
plot!(out_sp)

summaryplot(out)
out_sp.sample[1:100]

@btime out=mcmc_sampler(
    model_f, 
    lag=1, 
    init=model.mode, 
    desired_samples=2000,
    burn_in=0
)

plot(out)
summaryplot(out)
S
out.sample

tmp = [1,2,1,2,1,4]
V = 1:10

p = zeros(length(V))
p .+= [i ∈ tmp for i in V]
p ./= sum(p)