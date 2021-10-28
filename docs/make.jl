using Pkg
Pkg.activate("C:/users/boltg/.julia/enviroments/doc_writing")

using Documenter, InteractionNetworkModels

makedocs(
    sitename="My Documentation",
    format = Documenter.HTML(prettyurls = false),
    pages=["index.md", "model_sampling.md", "examples.md"] )