using InteractionNetworkModels, Plots, StatsPlots, LinearAlgebra

S = [[1,2],[2,1],[1,2],[2,3],[3,4],[1,2],[2,1]]

K_in_ub = 9
K_out_ub = 19
K_inner = DimensionRange(2,K_in_ub)
K_outer = DimensionRange(1,K_out_ub)
γ_s = 3.8 
γ_m = 3.0
d_s = FastEditDistance(FastLCS(20), 20)
d_m = MatchingDist(FastLCS(20))

V = 1:9
sis_model = SIS(S, γ_s, d_s, V, K_inner, K_outer)
sim_model = SIM(S, γ_m, d_m, V, K_inner, K_outer)

sis_mcmc = SisMcmcInsertDelete(
    ν_ed=3, ν_td=1, β=0.7, 
    len_dist=TrGeometric(0.8,1,K_in_ub),
    burn_in=2000, lag=50, desired_samples=500 
)

sim_mcmc = SimMcmcInsertDelete(
    ν_ed=3, ν_td=1, β=0.7, 
    len_dist=TrGeometric(0.8,1,K_in_ub),
    burn_in=2000, lag=100, desired_samples=500
)

sis_out = sis_mcmc(sis_model)
sim_out = sim_mcmc(sim_model)

plot(sis_out)
plot!(sim_out)

summaryplot(sis_out)
summaryplot!(sim_out)


# Order info 

order_inf_sis = [d_s(x,y)==0.0 ? 1.0 : d_m(x,y)/d_s(x,y) for x in sis_out.sample, y in sis_out.sample]
order_inf_sis = order_inf_sis[tril!(trues(size(order_inf_sis)),-1)]

order_inf_sim = [d_s(x,y)==0.0 ? 1.0 : d_m(x,y)/d_s(x,y) for x in sim_out.sample, y in sim_out.sample]
order_inf_sim = order_inf_sim[tril!(trues(size(order_inf_sim)),-1)]

density(order_inf_sis)
density!(order_inf_sim)
