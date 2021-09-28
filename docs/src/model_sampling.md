# Model Sampling 

```@docs
SisMcmcInitialiser
```

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
mcmc_sampler_gibbs = SisMcmcInsertDeleteGibbs(
    path_proposal, 
    K=100, 
    ν_gibbs=1, ν_trans_dim=1, β=0.6,
    init=SisInitRandEdit(5)
    )

```

And call it 

## Defining Models 

## Defining Samplers 

## Plotting 
