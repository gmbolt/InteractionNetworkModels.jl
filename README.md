# InteractionNetworkModels

This package contains supporting code of Bolt et al. (2021) *Temporary reference*. Features include 
1. Model types for Spherical Interaction Sequence (SIS) and Spherical Interaction Multisets (SIM) models, and the Hollywood model of Crane et al. (2018);
2. Samplers for models;
3. Samplers SIS/SIM posterior distributions.

## Installation Note

This package contains `PythonOT.jl` in its dependencies, which itself calls python code (specifically, the Python Optimal Transport (POT) library). Thus one will clearly need a Python installation for this package to work. Thankfully, `PythonOT.jl` will sort this for you, installing python in a Julia-specific folder by making use of the `Conda.jl` package. However, the `Conda.jl` package defaults to python version 3.9, and at the moment of writing this led to issues. If this is the case for you then a solution is to specify `Conda.jl` to use Python 3.8, the approach of which we now outline. 

For now, you can do the following in your root Julia enviroment. If you have never installed `Conda.jl` then you can skip the next comment a move to the steps bellow. If you *have* installed `Conda.jl` before then call `Pkg.rm("Conda")` to remove it and follow this with `Pkg.gc()`. Now, do the following 
1. In Julia REPL run `ENV["CONDA_JL_VERSION"]="3.8"` 
2. Install `PyCall.jl` via `Pkg.add("PyCall")`
3. Install `Conda.jl` via `Pkg.add("Conda")`
4. Build `PyCall` via `Pkg.build("PyCall")`

When we did these steps above the package seemd to install fine. 

## Data Structures

We represent a path with a vector of `Int` or `String` values. That is, if `x` is to store a path then we can have 
1. `x::Vector{Int}` - when we denote vertices with integers
2. `x::Vector{String}` - when we denote vertices with characters  

for example we might have `x=[1,2,1,2]` or `x=["a", "b", "a", "b"]`. 

We then represent an interaction sequence with vector of vectors of `Int` or `String`, that is, if `S` denotes our interaction sequence, we can have 
1. `S::Vector{Vector{Int}}`
2. `S::Vector{Vector{String}}`

for example, we might have `S = [[1,2], [3,4,1], [5,1,2]]` or `S = [["a", "b", "a"], ["c", "d"], ["a", "b", "c","a"]]`.

For readability we define the following aliases 
1. `Path{T} = Vector{T}` for `T = Int` and `T=String`
2. `InteractionSequence{T} = Vector{Vector{T}}` for `T = Int` and `T=String`

that is, we can use `Path{T}` and `InteractionSequence{T}` as we would `Vector{T}` or `Vector{Vector{T}}`. This is purely out of convenience. 

Note there is no separate representation of interaction multisets. The reason being this seemed an unnecessary complication, since one can simply interpret two interaction sequences as equal up a permutation of interaction order. Furthermore, one can can make use of the `Multisets.jl` package to convert any interaction sequence into an interaction multiset. For example, one could convert any vector `x::Vector{InteractionSequence{T}}` to type `Vector{Multiset{Path{T}}}` via the following

```julia 
using Multisets 
x = Multiset.(x) 
```


## Distance Metrics

A key feature of the models proposed in Bolt et al. (2021) are distance metrics. This includes distances between
1. Interactions, that is, paths;
2. Interaction sequences 
3. Interaction multisets 

We have defined some custom types to denote the various distance metrics. These have all been made subtypes of the `Metric` type of the `Distances.jl` package. 

### Interaction Distances 

Here we introduce the abstract type `InteractionDistance`, intended to cover distances between any type of interaction (e.g. if perhaps you would like to extend this beyond interactions being paths). We then have a further abstract subtype `PathDistance<:InteractionDistance`, which is to cover specifically distances between paths. Current supported path distances are a follows 

* *Longest Common Subsequence (LCS) Distance* - instantiated via `LCS()`
* *Longest Common Subpath (LCP) Distance* - instantiated via `LCP()`
* *Normalised LCS Distance* - instantiated via `NormLCS()`
* *Normalised LCP Distance* - instantiated via `NormLCP()`

Once a distance has been defined it can be called naturally as a function on two suitable arguments. For example, the following would evaluate the LCS distance between `x` and `y`
```julia
d = LCS()
x = [1,2,1,2]
y = [1,3,1,3]
d(x,y)
```

### Interaction Sequence Distances 

### Interaction Multiset Distances 


## Defining and Sampling From Models

Model types have been constructed to define models discussed in the paper. These include (i) Spherical Path Family (SPF) distributions (ii) Spherical Interaction Sequence (SIS) family distributions (iii) Spherical Interaction Multiset (SIM) family distributions. 

These are instantiated as follows 
* **Spherical Path Family (SPF)** - `SPF()`
    ```Julia
    Iᵐ = [1,2,1,2,1,2,1] # Modal path (interaction)
    d = LCS() # distance metric
    γ = 3.9 # Dispersion
    𝒱 = collect(1:10) # Vertex set
    K = 20 # Maximum path length

    # Define model 
    model = SPF(Iᵐ, γ, d, 𝒱, K=K)
    ```

## Posterior Inference

### Defining Posterior Distributions 

### Samplers 