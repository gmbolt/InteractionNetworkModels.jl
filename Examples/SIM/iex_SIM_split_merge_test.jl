using InteractionNetworkModels, StructuredDistances, Plots, Distributions

# Testing sampler
# ---------------

E = [[1,2,1,2],
    [1,2,1],
    [3,4,3], 
    [3,4], 
    [1,2], 
    [1,2,1],
    [1,2,3],
    [4,5],
    [7,8]]
d = MatchingDistance(FastLCS(101))
K_inner, K_outer = (DimensionRange(1,10), DimensionRange(1,25))
model = SIM(
    E, 3.5, 
    d,
    1:10,
    K_inner, K_outer
)


mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=1, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.8, 1, model.K_inner.u),
    burn_in=2000, lag=75, init=InitRandIns(10)
)

@time test = mcmc_sampler(
    model, desired_samples=5000, burn_in=0, lag=1
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
summaryplot(mcmc_out)

data = mcmc_out.sample
E_prior = SIM(E, 0.1, model.dist, model.V, model.K_inner, model.K_outer)
γ_prior = Uniform(0.5,7.0)

posterior = SimPosterior(data, E_prior, γ_prior)

# Construct posterior sampler
posterior_sampler = SimIexSplitMerge(
    mcmc_sampler,
    len_dist=TrGeometric(0.9,K_inner.l,K_inner.u),
    ν_ed=1, ν_td=1,
    β=0.7, ε=0.3
)

# Mode Conditional

E_init = [[1,2,1,3,4,3],[7,8,4,5,1,2,1],[10,1]]
@time posterior_out = posterior_sampler(
    posterior,
    # 4.9,
    desired_samples=100, lag=1, burn_in=0,
    S_init=E_init, γ_init=2.8,
    aux_init_at_prev=true,
);
plot(
    posterior_out, E, size=(600,400),   
    xlabel=["" "Sample Index"], 
    ylabel=["Distance to True Mode" "γ"],
    yguidefontsize=[9 9],
    xguidefontsize=9
)
posterior_out.S_sample

posterior_out
# Testing iMCMC proposal functions 
# --------------------------------

aux_mcmc = SimMcmcInsertDelete()
mcmc = SimIexSplitMerge(aux_mcmc)

# Standard proposal generation functions

# Split
S_curr = [rand(1:3, 3) for i in 1:5]
S_prop = deepcopy(S_curr)

imcmc_multi_split_prop_sample!(S_curr, S_prop, mcmc)

S_curr
S_prop

# Merge 
S_curr = [rand(1:3, 3) for i in 1:5]
S_prop = deepcopy(S_curr)

imcmc_multi_merge_prop_sample!(S_curr, S_prop, mcmc)

S_curr
S_prop


# When we have length one paths 
S_curr = [rand(1:3, 1) for i in 1:5]
push!(S_curr, [1,2])
push!(S_curr,[1,2,3,2])
S_prop = deepcopy(S_curr)

imcmc_special_multi_split_prop_sample!(S_curr, S_prop, mcmc)

S_curr
S_prop

S_curr = deepcopy(S_prop)

imcmc_special_multi_merge_prop_sample!(S_curr, S_prop, mcmc)

S_curr
S_prop


