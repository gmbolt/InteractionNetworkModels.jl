using InteractionNetworkModels, BenchmarkTools, Plots, Distributions, StatsPlots
using Distances, Distances, ProgressMeter

d = Hamming()
V = 20
mode = rand(Bool,V,V)
model = CER(
    mode, 0.05
)
n = 10
data = draw_sample(model, n)

plot(map(x->d(x,mode), data))

G_prior = CER(mode, 0.5)
α_prior = ScaledBeta(1,10,0,0.5)
plot(α_prior)

posterior = CerPosterior(data, G_prior, α_prior)

mcmc = CerPosteriorMcmc(τ=0.001, ε=0.01)

@time out = mcmc(
    posterior, 
    desired_samples=10000, burn_in=0, lag=1,
    G_init=rand_perturb(mode, 0.001),
    α_init=0.02
)
acceptance_prob(mcmc)
plot(out, mode)

plot(log_posterior_prob(out))

hline!([model.α], subplot=2)

# Verify alpha is unbiased 
function sim(
    true_model::CER, 
    mcmc::CerPosteriorMcmc,
    n_train::Int,
    n_reps::Int
    )

    # α_true = true_model.α
    α_ests = Float64[]
    @showprogress for i in 1:n_reps
        data = draw_sample(true_model, n_train)
        G_prior = CER(true_model.mode, 0.5)
        α_prior = ScaledBeta(1,10,0,0.5)
        posterior = CerPosterior(data, G_prior, α_prior)
        out = mcmc(
            posterior, 
            G_init=rand_perturb(true_model.mode, 0.001),
            α_init=true_model.α+0.1,
            loading_bar=false
        )
        push!(α_ests, mean(out.α_sample))
    end 
    return α_ests
end 

mcmc = CerPosteriorMcmc(
    τ=0.001, ε=0.01, 
    desired_samples=250,
    burn_in=5_000, 
)
ests = sim(model, mcmc, 20, 500)
boxplot(ests)
hline!([model.α])
