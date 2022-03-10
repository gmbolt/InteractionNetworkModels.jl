using InteractionNetworkModels, Plots, Distributions
using StatsPlots, Plots.Measures, StatsBase
using Distances, StructuredDistances

V = 50
Sᵐ = [rand(1:V,rand(2:5)) for i in 1:10]
d = FastEditDistance(FastLCS(100), 100)
γ = 6.0
K_inner, K_outer = (DimensionRange(1,10), DimensionRange(1,25))
# K_inner, K_outer = (Inf,Inf)
model = SIS(
    Sᵐ, γ,
    d,
    1:V,
    K_inner, K_outer)

d(Sᵐ,Sᵐ)

mcmc_sampler = SisMcmcInsertDelete(
    ν_ed=1, ν_td=1, β=0.7, len_dist=TrGeometric(0.9,1,K_inner.u),
    lag=30, burn_in=1000,
    K=200
)

@time test = mcmc_sampler(
    model, desired_samples=50000, burn_in=0, lag=1)

plot(test)
summaryplot(test)
test.sample

@time mcmc_out = mcmc_sampler(
    model,
    desired_samples=100,
    lag=500,
    burn_in=10000
    )


plot(mcmc_out)
summaryplot(mcmc_out)

data = mcmc_out.sample
S_prior = SIS(Sᵐ, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,10.0)

posterior = SisPosterior(data, S_prior, γ_prior)

posterior_sampler = SisIexInsertDelete(
    mcmc_sampler,
    ν_ed=1, ν_td=1, len_dist=TrGeometric(0.8, 1, K_inner.u),
    ε=0.2, β=0.7,
    K=200,
    α=0.0,
    desired_samples=50, burn_in=0, lag=1
)


# Mode Conditional
S_init, ind = sample_frechet_mean(posterior.data, posterior.dist)
d(S_init, Sᵐ)
initialiser = InitRandEdit(2)

@time posterior_out = posterior_sampler(
    posterior,
    S_init=S_init, 
    γ_init=4.0,
    # aux_init_at_prev=true,
    desired_samples=100
);

posterior_out
posterior_out.S_sample
plot(posterior_out, Sᵐ)

# Gamma conditional

S_fix = Sᵐ
posterior_out = posterior_sampler(
    posterior,
    S_fix,
    desired_samples=500, lag=1, burn_in=0,
    γ_init=4.6
)

plot(posterior_out)

# Joint

using BenchmarkTools

@time posterior_out = posterior_sampler(
    posterior,
    desired_samples=500, lag=2, burn_in=1000,
    S_init = S_init, γ_init=6.0
)

posterior_out
posterior_out.S_sample
plot(posterior_out, Sᵐ)


# Predictives 
# -----------

# Distances to mode
dist_pred = DistanceToModePredictive(posterior_out)
outer_dim_pred = OuterDimensionPredictive(posterior_out)

pred_sample = draw_sample(mcmc_sampler, dist_pred, n_samples=500, n_reps=length(data))
obs_dists = map(x->d(x,model.mode), data)
density(pred_sample, alpha=0.1, label=nothing, lw=2)
density!(obs_dists, c=:red, lw=2, label="Observed")

# Outer dimension (length)
pred_sample = draw_sample(mcmc_sampler, outer_dim_pred, n_samples=500, n_reps=length(data))
obs_outer_dim = length.(data)
density(pred_sample, alpha=0.1, label=nothing, lw=2)
density!(obs_outer_dim, c=:red, lw=2, label="Observed")

# Mean inner length 
inner_pred = MeanInnerDimensionPredictive(posterior_out)
pred_sample = draw_sample(mcmc_sampler, inner_pred, n_samples=10, n_reps=50)


# Missing entry predictive
data_test = draw_sample(
    mcmc_sampler,
    model,
    desired_samples=1,
    lag=1,
    burn_in=10000
    )
S_test = data_test[1]
Sᵐ
posterior_pred = pred_missing(S_test, (1,1), posterior_out)
get_prediction(posterior_pred)

model_pred = pred_missing(S_test, (1,1), model)
get_prediction(model_pred)

# Checking flip proposal 
x = [1,3,4,5,6,7]
P, vmap, vmap_inv = get_informed_proposal_matrix(posterior, 0.0)
flip_informed_excl!(x,[1],P)
x
out = Int[]
for i in 1:1000
    flip_informed_excl!(x,[1],P)
    push!(out, x[1])
end 

counts(out)