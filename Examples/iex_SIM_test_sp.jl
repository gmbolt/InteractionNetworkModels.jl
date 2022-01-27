using InteractionNetworkModels, Plots, Distributions
using StatsPlots, Plots.Measures, StatsBase

E = [[1,2,1,2],
    [1,2,1],
    [3,4,3], 
    [3,4], 
    [1,2]]
d = MatchingDist(FastLSP(100))
K_inner, K_outer = (DimensionRange(2,50), DimensionRange(1,50))
model = SIM(
    E, 4.4, 
    d,
    1:50,
    K_inner, K_outer)


mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=4, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.9, 1, model.K_inner.u),
    burn_in=2000, lag=50
)

mcmc_sampler_sp = SimMcmcInsertDeleteSubpath(
    ν_ed=7, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.9, 1, model.K_inner.u),
    burn_in=3000, lag=50
)


@time test = mcmc_sampler(
    model, desired_samples=500)

plot(test)
summaryplot(test)
test.sample

@time mcmc_out = mcmc_sampler(
    model,
    desired_samples=50,
    lag=500,
    burn_in=10000
    )


plot(mcmc_out)
summaryplot(mcmc_out)

data = mcmc_out.sample
E_prior = SIM(E, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,7.0)

target = SimPosterior(data, E_prior, γ_prior)

# Construct posterior sampler
posterior_sampler = SimIexInsertDelete(
    mcmc_sampler,
    len_dist=TrGeometric(0.9,K_inner.l,K_inner.u),
    ν_ed=1, ν_td=1,
    β=0.7, ε=0.3
)

# Mode Conditional
E_init, ind = sample_frechet_mean(target.data, target.dist)
d(E_init, E)

@time posterior_out = posterior_sampler(
    target,
    desired_samples=4000, lag=1, burn_in=0,
    S_init=E_init[3:3], γ_init=4.5
);

posterior_out.S_sample
plot(posterior_out, E)
plot(length.(posterior_out.S_sample))
findall(diff(length.(posterior_out.S_sample)).>0)

plot(posterior_out.suff_stat_trace)

S_est = deepcopy(posterior_out.S_sample[end])

mapreduce(x->d_lsp(x,S_est), +, data)
mapreduce(x->d_lsp(x,E), +, data)

plot(length.(data))

i=1
tmp1, C1 = d_lsp(E, data[i])
tmp2, C2 = d_lsp(data[i], E)
C1
C2


pairwise(LSP(), E, data[i])
pairwise(LSP(), data[i], E)
d_lsp(S_est, data[i])

using Hungarian
data[1]
LSP()(data[i][1], E[1])
LSP()(E[1], data[i][1])
FastLSP(100)(data[i][1], E[1])
FastLSP(100)(E[1], data[i][1])

E[1]
data[i][1]


[d_lsp(x,E) for x in data]

d_lsp(E, S_est)
d_lsp(S_est, E)
tmp, C = d_lsp(S_est, E)
C
S_est
E


d([[1]], [[2]])
d([[2]],[[1]])

d_lcs = MatchingDist(LCS())

d_lcs(E, S_est)
d_lcs(S_est, E)

map(x->d(x,E), data)
E
d

plot(map(x->d_lsp(E,x), data))
plot!(map(x->d_lsp(x,S_est), data))
S_est

print_matching(d, S, data[1])


print_map_est(posterior_out)

plot(posterior_out, E)

d([[1,1,1],[2,2]], [[2,1]])

# Dispersion conditional
E_fix = Eᵐ
posterior_out = posterior_sampler(
    target, 
    E_fix, 
    desired_samples=1000, lag=1, burn_in=0,
    init=4.6
)
plot(posterior_out)

