var documenterSearchIndex = {"docs":
[{"location":"examples.html#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples.html#Model-Sampling","page":"Examples","title":"Model Sampling","text":"","category":"section"},{"location":"examples.html","page":"Examples","title":"Examples","text":"using InteractionNetworkModels, Distributions \r\nmodel_mode = Hollywood(-3.0, Poisson(7), 10)\r\nS = sample(model_mode, 10)\r\nV = collect(1:10)\r\nd = FastGED(FastLCS(21),21)\r\nmodel = SIS(S, 5.5, d, V, 20, 20)","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"Now we define sampler ","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"path_proposal = PathPseudoUniform(model.V, TrGeometric(0.5, 1, model.K_inner))\r\nmcmc_sampler_gibbs = SisMcmcInsertDeleteGibbs(\r\n    path_proposal, \r\n    K=100, \r\n    ν_gibbs=1, ν_trans_dim=1, β=0.6,\r\n    init=SisInitRandEdit(5)\r\n    )\r\n","category":"page"},{"location":"examples.html","page":"Examples","title":"Examples","text":"And call it ","category":"page"},{"location":"model_sampling.html#Model-Sampling","page":"Model Sampling","title":"Model Sampling","text":"","category":"section"},{"location":"model_sampling.html","page":"Model Sampling","title":"Model Sampling","text":"SisMcmcInitialiser","category":"page"},{"location":"model_sampling.html#InteractionNetworkModels.SisMcmcInitialiser","page":"Model Sampling","title":"InteractionNetworkModels.SisMcmcInitialiser","text":"Abstract type representing initialisation schemes for SIS model samplers. \n\n\n\n\n\n","category":"type"},{"location":"model_sampling.html#Defining-Models","page":"Model Sampling","title":"Defining Models","text":"","category":"section"},{"location":"model_sampling.html#Defining-Samplers","page":"Model Sampling","title":"Defining Samplers","text":"","category":"section"},{"location":"model_sampling.html#Plotting","page":"Model Sampling","title":"Plotting","text":"","category":"section"},{"location":"index.html#Overview","page":"Overview","title":"Overview","text":"","category":"section"},{"location":"index.html","page":"Overview","title":"Overview","text":"Who is this package for and what can it do. ","category":"page"}]
}
