# Examples 

## Model Sampling 

```@example model_sampling
using InteractionNetworkModels, Distributions 
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
V = collect(1:10)
d = FastGED(FastLCS(21),21)
model = SIS(S, 5.5, d, V, 20, 20)
```

Now we define sampler 

```@example model_sampling
path_proposal = PathPseudoUniform(model.V, TrGeometric(0.5, 1, model.K_inner))
mcmc_sampler = SisMcmcInsertDelete(
    path_proposal, 
    K=model.K_inner, 
    ν_ed=1, ν_td=1, β=0.7
    )

```

And call it 

```@example model_sampling
using Plots
mcmc_out = mcmc_sampler(model)
```

And plot it 

```@example model_sampling
plot(mcmc_out)
```

...increase lag and burn-in for better samples...


```@example model_sampling
mcmc_out = mcmc_sampler(
    model, 
    desired_samples=1000, 
    burn_in=1000, lag=50
    )
plot(mcmc_out)
```