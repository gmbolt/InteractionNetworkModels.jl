# InteractionNetworkModels

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
 

## Distance Metrics

We have defined some custom types to denote different distance metrics, both between interactions and interaction sequences. These have been made subtypes of the `Metric` type of the `Distances.jl` package. 

## Models

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

## Posterior Distributions

## MCMC Samplers 

### Models 

### Posteriors