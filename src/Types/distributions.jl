export ScaledBeta, TrGeometric, TrBinomial, StringCategorical, likelihood, log_likelihood
export normalising_const
## Scaled beta distribution
"""
Scaled Beta distribution. Instantiate with ScaledBeta() e.g.
```
d = ScaledBeta(α, β, 0.0, 10.0)
```
would make a ScaledBeta type d which is bounded by 0.0 and 10.0. Note
- (α,β) are usual parameters;
- lb = lower bound;
- ub = upper bound.
"""
struct ScaledBeta<:ContinuousUnivariateDistribution
    α::Float64
    β::Float64
    lb::Float64
    ub::Float64
end

Distributions.rand(d::ScaledBeta) = d.lb + (d.ub-d.lb) * rand(Beta(d.α, d.β))
Distributions.mean(d::ScaledBeta) = d.lb + (d.ub-d.lb) * ( d.α / (d.α + d.β) )
Distributions.pdf(d::ScaledBeta, x::Real) = (d.lb ≤ x ≤ d.ub) ? pdf(Beta(d.α, d.β), (x - d.lb)/(d.ub-d.lb))/(d.ub-d.lb) : 0
Distributions.minimum(d::ScaledBeta) = d.lb
Distributions.maximum(d::ScaledBeta) = d.ub

## Trucated geometric
"""
`TrGeomtric(p::Float64, lb::Int, ub::Int)` 

Truncated Geometric distribution. Note:
- lb = lower bound = samllest possible value (default=0)
- ub = upper bound = largest possible value.
Instantiation:

1. `TrGeometric(p, ub)` -  here `lb` defaults to 0
2. `TrGeometric(p, lb, ub)` here `lb` is specified

"""
struct TrGeometric<:DiscreteUnivariateDistribution
    p::Float64 # Prob param
    lb::Int # Lower bound (smallest value)
    ub::Int  # Upper bound
end

function normalising_const(d::TrGeometric)
    return (1 - (1-d.p)^(d.ub-d.lb+1))
end 

TrGeometric(p::Float64, ub::Int) = TrGeometric(p, 0, ub)

function Distributions.pdf(d::TrGeometric, x::Integer)
    _insupport = insupport(d, x)
    return _insupport ? d.p*(1-d.p)^(x-d.lb) / normalising_const(d) : 0.0
end

function Distributions.logpdf(d::TrGeometric, x::Integer)
    _insupport = insupport(d, x)
    return _insupport ? log(d.p) + (x-d.lb) * log(1-d.p) -  log(normalising_const(d)) : -Inf
end

function Distributions.cdf(d::TrGeometric, x::Number)
    if d.lb ≤ x ≤ d.ub
        return ( 1 - (1 - d.p)^(trunc(Int,x)-d.lb+1) )/normalising_const(d)
    else
        return float(x > d.ub)
    end
end

# Here I make use of closed form of generalised inverse of the TrGeom dist.
# Not sure if this is the fastest implementation
function Distributions.quantile(d::TrGeometric, q::Real)
    return 0.0<q<1.0 ? ceil(Int, log(1-q*normalising_const(d))/log(1-d.p) - 1 )+d.lb : d.ub*(p≥1.0) + d.lb*(q≤0.0)
end

Distributions.minimum(d::TrGeometric) = d.lb
Distributions.maximum(d::TrGeometric) = d.ub

#Truncated Binomial wrappper - solves the problem of when we have n=0

struct TrBinomial <: DiscreteUnivariateDistribution
    n::Int
    p::Real
    lb::Int
    ub::Int
    dist::Distribution
    function TrBinomial(n::Int, p::Real, lb::Int, ub::Int)
        @assert (lb ≤ ub) "Lower bound must be ≤ upper bound."
        if lb < ub 
            tmp_dist = Truncated(Binomial(n, p), lb, ub)
        else 
            tmp_dist = DiscreteUniform(lb, ub)
        end 
        new(n, p, lb, ub, tmp_dist)
    end
end 

function Distributions.rand(d::TrBinomial)
    return rand(d.dist)
end
function Distributions.rand(d::TrBinomial, n::Int)
    return rand(d.dist, n)
end 


struct StringCategorical <: DiscreteUnivariateDistribution
    x::Vector{String}
    p::Vector{T} where T<:Real
    latent_dist::Distribution
    probability_map::Dict{String, Float64}
    function StringCategorical(x::Vector{String}, p::Vector{T} where T<: Real)
        latent_dist = Categorical(p)
        probability_map = Dict(val => prob for (val, prob) in zip(x, p))
        new(x, p, latent_dist, probability_map)
    end 
end 

Distributions.pdf(d::StringCategorical, val::String) = d.probability_map[val]
function likelihood(d::StringCategorical, val::Vector{String})
    z = 1.0
    for s in val
        z *= pdf(d, s)
    end 
    return z
end 

Distributions.logpdf(d::StringCategorical, val::String) = log(d.probability_map[val])
function log_likelihood(d::StringCategorical, val::Vector{String})
    z = 0.0
    for s in val
        z += logpdf(d, s)
    end 
    return z
end 


function Distributions.rand(d::StringCategorical)
    return d.x[rand(d.latent_dist)]
end 

function Distributions.rand(d::StringCategorical, n::Int)
    return d.x[rand(d.latent_dist, n)]
end 
