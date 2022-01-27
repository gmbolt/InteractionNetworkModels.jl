using InteractionNetworkModels, Plots, Distributions
using StatsPlots, Plots.Measures, StatsBase, LaTeXStrings

E = [[1,2,1,2],
    [1,2,1],
    [3,4,3], 
    [3,4], 
    [1,2]
]

d = MatchingDist(FastLCS(100))

K_inner = DimensionRange(1,50)
K_outer = DimensionRange(1,50)
V = 1:30
model = SIM(
    E, 4.4, 
    d,
    V, K_inner, K_outer)


mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=7, ν_td=1, β=0.7,
    len_dist=TrGeometric(0.9, 1, K_inner.u),
    burn_in=2000, lag=50
)

@time test = mcmc_sampler(
    model, desired_samples=500)

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

target = SimPosterior(data, E_prior, γ_prior)

# Construct posterior sampler
posterior_sampler = SimIexInsertDelete(
    mcmc_sampler,
    len_dist=TrGeometric(0.9,1,100),
    ν_ed=1, ν_td=1,
    β=0.7, ε=0.2
)


# Dispersion conditional

E_restr = E[1:2]
posterior_out = posterior_sampler(
    target,  
    desired_samples=200, lag=1, burn_in=0,
    γ_init=4.6, S_init=E[1:1]
)
posterior_out.S_sample
plot(posterior_out)


# Nested means 

x_num_inters = [zeros(200) for i in eachindex(E)]

for i in eachindex(E)
    draw_sample_gamma!(
        x[i],
        posterior_sampler, 
        target, 
        E[1:i], 
        burn_in=250, lag=5, 
        γ_init = 4.4    
    )
end 


density(
    x, 
    label=[L"\mathcal{E}_{1:%$(i)}" for j=1:1, i=eachindex(E)],
    legend_font_pointsize=10,
    legend_font_valign=:bottom,
    xlabel="γ",
    ylabel="(Conditional) Posterior Density"
)

# Growing dimensions 

modes = InteractionSequenceSample{Int}()
for i in eachindex(x)
    tmp_count = 0 
    path_ind = findlast(cumsum(length.(E)) .< i)
    if path_ind == nothing 
        E_restr = [E[1][1:i]]
    else 
        entry_ind = i - cumsum(length.(E))[path_ind]
        E_restr = E[1:path_ind]
        push!(E_restr, E[path_ind+1][1:entry_ind])
    end 
    push!(modes, E_restr)
end 
x = [zeros(200) for i in eachindex(modes)]
for i in eachindex(modes)
    draw_sample_gamma!(
        x[i],
        posterior_sampler, 
        target, 
        modes[i], 
        burn_in=250, lag=5, 
        γ_init = 4.4    
    )
end 

density(
    x, 
    label=["Total dim. = $(i)" for j=1:1, i=eachindex(x)],
    legend=:outerright,
    xlabel="γ",
    ylabel="(Conditional) Posterior Density"
)


# Using dependent sampler 
posterior_sampler_dep = SimIexInsertDeleteDependent(
    mcmc_sampler,
    len_dist=TrGeometric(0.9,1,100),
    ν_ed=1, ν_td=1,
    β=0.7, ε_ed=0.07, ε_td=0.2
)

posterior_out = posterior_sampler_dep(
    target, 
    desired_samples=400, lag=1, burn_in=0,
    γ_init=4.6, S_init=E[1:1]
)
plot(posterior_out, E)
posterior_out.S_sample
