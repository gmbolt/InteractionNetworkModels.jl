using InteractionNetworkModels, Distributions, Distances, StructuredDistances
using BenchmarkTools, StatsBase, Plots, StatsPlots

V = 50
Sᵐ = [rand(1:V,rand(2:5)) for i in 1:10]
d = FastEditDistance(FastLCS(100), 100)
γ = 3.0
K_inner, K_outer = (DimensionRange(1,10), DimensionRange(1,25))
# K_inner, K_outer = (Inf,Inf)
model = SIS(
    Sᵐ, γ,
    d,
    1:V,
    K_inner, K_outer
)

aux_mcmc = SisMcmcInsertDelete(
    ν_ed=1, ν_td=1, β=0.7, len_dist=TrGeometric(0.9,1,K_inner.u),
    lag=1, burn_in=0,
    K=200
)

S_curr = deepcopy(Sᵐ)
S_prop = S_curr[1:(end-1)]

aux_eval = SisAuxTermEvaluator(aux_mcmc, d, 1:V, K_inner, K_outer)
initialise!(aux_eval, S_prop)

n = 100
aux_data = [[Int[]] for i in 1:n]

@time aux_term_current(
    S_curr, S_prop,
    γ, d, 1:V, K_inner, K_outer, 
    aux_mcmc, aux_data
)

@time aux_term_new(
    S_curr, S_prop,
    γ, 
    n, 
    aux_eval
)

plot(map(x->d(x, S_prop), y), label="New")
plot!(map(x->d(x, S_prop), aux_data), label="Old")

aux_eval.S_curr


aux_mcmc.dist_curr

@btime aux_term_current(
    $S_curr, $S_prop,
    $γ, $d, 1:$V, $K_inner, $K_outer, 
    $aux_mcmc, $aux_data
)

@btime aux_term_new(
    $S_curr, $S_prop,
    $γ, 
    $n, 
    $aux_eval
)

plot(map(x->d(x, S_prop), y), label="New")
plot!(map(x->d(x, S_prop), aux_data), label="Old")


m = 100
current = [aux_term_current(
                S_curr, S_prop,
                γ, d, 1:V, K_inner, K_outer, 
                aux_mcmc, aux_data
            ) for i in 1:m
];
new = [aux_term_new(
            S_curr, S_prop,
            γ, 
            n, 
            aux_eval
        ) for i in 1:m
];

dotplot([current, new])