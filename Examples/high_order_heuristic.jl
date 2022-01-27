using InteractionNetworkModels, Distributions

# On uniform data 
# ---------------

S = [rand(1:4,rand(5:10)) for i in 1:7]

@time get_subseqs(S, 3)

data = [
    [rand(1:4,rand(5:30)) for i in 1:4] for i in 1:10
]

get_subseq_counts(data, 3)

x = get_subseq_proportions(data, 3)

dict_rank(x)


# On SIM model data 
# -----------------

# The Model(s)

S = [[1,1], [1,2], [1,2,3], [1,4], [2,3]]
V = 1:13
d = MatchingDist(FastLCS(100))
model = SIM(S, 4.0, d, V, 50, 50)

mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=6, β=0.6, ν_td=1,  
    len_dist=TrGeometric(0.8, 1, model.K_inner.u),
    lag=1,
    K=200)

@time out=mcmc_sampler(
    model, 
    init=model.mode, 
    desired_samples=100,
    burn_in=5000, lag=300
)

summaryplot(out)
out.sample
subseq_props = get_subseq_proportions(out.sample, 3)
dict_rank(subseq_props)

out.sample