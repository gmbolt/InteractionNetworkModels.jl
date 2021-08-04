using Distributions, StatsBase

struct Hollywood 
    α::Real 
    θ::Real 
    ν::DiscreteUnivariateDistribution
    K::Real
    function Hollywood(α::Real, θ::Real, ν::DiscreteUnivariateDistribution, K::Int)
        @assert ((0 < α < 1.0) & (θ > -α)) | ((α < 0) & (θ = - K * α)) "Check parameters satisfy required constraints"
        new(α, θ, ν, K) 
    end     
end 

Hollywood(α::Real, θ::Real, ν::DiscreteUnivariateDistribution) = Hollywood(α, θ, ν, Inf)

function StatsBase.sample!(
    out::InteractionSequence,
    model::Hollywood
)


end 