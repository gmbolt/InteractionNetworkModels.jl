### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 80b90b23-c999-4836-ae83-d1b7017daa46
begin
	using Pkg
	Pkg.activate(joinpath(Base.current_project(), "..", ".."))  # Tells Pluto.jl we are going to use local enviroment 
	using InteractionNetworkModels, Distributions, Plots
end 

# ╔═╡ 0439d40e-768e-44d0-85bc-417b063f4315
md"
# Overview

Here we illustrate how one can sample from the Spherical Interaction Sequence (SIS) models. This amounts to the following steps 
1. Define an `model::SIS` model object, specifying the required parameters 
2. Construct an `mcmc::SisMcmcSampler` object
3. Call `mcmc(model)` to generate samples 
4. Explore outputs manually or via pre-defined visualisations

"

# ╔═╡ 8dda732d-6cd5-4ac9-a749-277b1ba49cea
md"
...constructing model...
"

# ╔═╡ 09b424d8-b54d-4d12-9367-c0413c46475e
model = SIS(
	[[1,2],[1,2],[2,3,4],[2,3,4]],
	5.0, 
	FastEditDistance(FastLCS(100),100),
	1:10, 
	30, 30
);

# ╔═╡ 48ad33b4-b44c-482e-aa98-baa5a3e890ce
md"
...constructing MCMC sampler...
"

# ╔═╡ 5f990069-ba98-4930-9806-3ed8c25e7253
path_dist = PathPseudoUniform(model.V,TrGeometric(0.8, 1, 10));

# ╔═╡ 699e9669-b105-4d5a-962b-7bf63acb8b49
mcmc = SisMcmcInsertDelete(path_dist);

# ╔═╡ 1504fd48-9e5e-4536-a0ed-7ebc26126b92
md"
...now to get a sample we can use `mcmc` like a function, that is, we can call `mcmc(model)` to obtain an MCMC sample from `model`...
"

# ╔═╡ 508ba9f3-fc56-4012-a447-8bb6c675f271
x = mcmc(model, desired_samples=5000, lag=5, burn_in=1000)

# ╔═╡ ef5fa55b-198c-41a6-acb4-e0bb964c822a
md"...we can now visualise a trace plot by simply calling `plot(x)` where `x::SisMcmcOutput` is the output from our previous call of the sampler...
"

# ╔═╡ ae0503ce-111e-435a-9f6a-a1010d707880
plot(x)

# ╔═╡ Cell order:
# ╠═80b90b23-c999-4836-ae83-d1b7017daa46
# ╟─0439d40e-768e-44d0-85bc-417b063f4315
# ╟─8dda732d-6cd5-4ac9-a749-277b1ba49cea
# ╠═09b424d8-b54d-4d12-9367-c0413c46475e
# ╟─48ad33b4-b44c-482e-aa98-baa5a3e890ce
# ╠═5f990069-ba98-4930-9806-3ed8c25e7253
# ╠═699e9669-b105-4d5a-962b-7bf63acb8b49
# ╟─1504fd48-9e5e-4536-a0ed-7ebc26126b92
# ╠═508ba9f3-fc56-4012-a447-8bb6c675f271
# ╟─ef5fa55b-198c-41a6-acb4-e0bb964c822a
# ╠═ae0503ce-111e-435a-9f6a-a1010d707880
