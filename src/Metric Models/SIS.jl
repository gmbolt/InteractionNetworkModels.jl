using Distributions, Distances
export SIS, SisPosterior 

struct DimensionRange
    l::Real 
    u::Real 
end 
# NOTE - both these functions perform worse if called repeatedly.
# If we have r::DimensionRange better to assign values such as..
# l, u = (r.l, r.u)
# and then use these in the code. This is what I have done in MCMC samplers. 
function isin(
    r::DimensionRange,
    val::Real
    )
    return (val >= r.l) & (val <= r.u)
end 

function notin(
    r::DimensionRange,
    val::Real
    )
    return (val < r.l) | (val > r.u)
end 

struct SIS
    mode::Vector{Path{Int}} # Mode
    γ::Real # Precision
    dist::Metric # Distance metric
    V::UnitRange # Vertex Set
    K_inner::DimensionRange # Maximum interaction sequence size
    K_outer::DimensionRange # Maximum path (interaction) length
end

SIS(
    mode::InteractionSequence{Int}, 
    γ::Real, 
    dist::Metric, 
    V::UnitRange
) = SIS(
        mode, γ, dist, V, 
        DimensionRange(1,Inf), 
        DimensionRange(1,Inf)
)

SIS(
    mode::InteractionSequence{Int}, 
    γ::Real, 
    dist::Metric, 
    V::UnitRange,
    K_inner::Real, K_outer::Real
) = SIS(
        mode, γ, dist, V, 
        DimensionRange(1,K_inner), 
        DimensionRange(1,K_outer)
)

function Base.show(
    io::IO, model::SIS
    ) 

    title = "SIS Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(model))
        println(io, par, " = $(getfield(model, par))")
    end 
end 


struct SisPosterior
    data::InteractionSequenceSample{Int}
    S_prior::SIS
    γ_prior::ContinuousUnivariateDistribution
    dist::Metric
    V::AbstractArray{Int}
    K_inner::DimensionRange
    K_outer::DimensionRange
    sample_size::Int
    function SisPosterior(
        data::InteractionSequenceSample{Int},
        S_prior::SIS, 
        γ_prior::ContinuousUnivariateDistribution
        ) 

        dist = S_prior.dist
        V = S_prior.V
        K_inner = S_prior.K_inner
        K_outer = S_prior.K_outer
        sample_size = length(data)
        new(data, S_prior, γ_prior, dist, V, K_inner, K_outer, sample_size)
    end 
end 

