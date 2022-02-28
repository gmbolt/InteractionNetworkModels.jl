using InteractionNetworkModels, Plots, Distributions
using StatsPlots, Plots.Measures, StatsBase

E = [[1,2,1,2],
    [1,2,1,1],
    [3,4,3,4], 
    [3,4], 
    [1,2], 
    [1,2,1]
    ]
d = AvgSizeMatchingDist(FastLCS(101), 0.8)
K_inner, K_outer = (DimensionRange(1,100), DimensionRange(1,25))
model = SIM(
    E, 3.0, 
    d,
    1:10,
    K_inner, K_outer)


mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=4, ν_td=3, β=0.7,
    len_dist=truncated(Poisson(mean(length, model.mode)), K_inner.l, K_inner.u),
    burn_in=2000, lag=50
)

mcmc_sampler_prop = SimMcmcInsertDeleteProportional(
    ν_ed=1, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.8, 1, model.K_inner.u),
    burn_in=2000, lag=75, init=InitRandIns(10)
)

@time test = mcmc_sampler(
    model, desired_samples=5000, burn_in=0, lag=1
)
plot(test)
summaryplot(test)
test.sample

@time test = mcmc_sampler_prop(
    model, desired_samples=5000, burn_in=0, lag=1
)
plot(test)
summaryplot(test)

test.sample
@time test = mcmc_sampler(
    model, desired_samples=500
)
plot(test)
summaryplot(test)

@time mcmc_out = mcmc_sampler(
    model,
    desired_samples=50,
    lag=500,
    burn_in=10000
    )


plot(mcmc_out)
mcmc_out.sample
summaryplot(mcmc_out)

data = mcmc_out.sample
E_prior = SIM(E, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,7.0)

posterior = SimPosterior(data, E_prior, γ_prior)

# Construct posterior sampler
posterior_sampler = SimIexInsertDelete(
    mcmc_sampler,
    len_dist=TrGeometric(0.9,K_inner.l,K_inner.u),
    ν_ed=1, ν_td=1,
    β=0.7, ε=0.3
)

posterior_sampler_prop = SimIexInsertDeleteProportional(
    mcmc_sampler_prop,
    len_dist=TrGeometric(0.9,K_inner.l,K_inner.u),
    ν_ed=1, ν_td=1,
    β=0.7, ε=0.3
)

# Mode Conditional
E_init = sample_frechet_mean(posterior.data, posterior.dist)[1]
d(E_init, E)

# E_init = [[10,10]]
@time posterior_out = posterior_sampler(
    posterior,
    # 4.9,
    desired_samples=100, lag=1, burn_in=0,
    S_init=E_init[1:3], γ_init=2.8,
    aux_init_at_prev=true,
);

plot(posterior_out, E)
posterior_out.S_sample
plot(posterior_out, E)
plot(length.(posterior_out.S_sample))
findall(diff(length.(posterior_out.S_sample)).>0)

plot(length.(posterior_out.S_sample))
print_map_est(posterior_out)

plot(posterior_out, E)

E_init[1] = [1,2,1,4,4,4,4,4,4,4,4,4,4,1,2]
@time posterior_out = posterior_sampler(
    posterior,
    # 4.9,
    desired_samples=1000, lag=1, burn_in=0,
    S_init=E_init, γ_init=2.8,
    aux_init_at_prev=true,
);
plot(posterior_out, E)
posterior_out.S_sample
@time posterior_out_prop = posterior_sampler_prop(
    posterior,
    # 4.9,
    desired_samples=1000, lag=1, burn_in=0,
    S_init=E_init, γ_init=2.8,
    aux_init_at_prev=true,
);
plot(posterior_out_prop, E)
posterior_out_prop.S_sample

posterior_out.S_sample
# Dispersion conditional
E_fix = Eᵐ
posterior_out = posterior_sampler(
    target, 
    E_fix, 
    desired_samples=1000, lag=1, burn_in=0,
    init=4.6
)
plot(posterior_out)

