export SpfPosteriorMcmcOutput, SpfPosteriorModeConditionalMcmcOutput, SpfPosteriorDispersionConditionalMcmcOutput
export SisPosteriorMcmcOutput, SisPosteriorModeConditionalMcmcOutput, SisPosteriorDispersionConditionalMcmcOutput
export SimPosteriorModeConditionalMcmcOutput, SimPosteriorDispersionConditionalMcmcOutput

# ==========
#    SPF 
# ==========

struct SpfPosteriorMcmcOutput{T<:Union{Int, String}}
    I_sample::Vector{Path{T}}
    γ_sample::Vector{Float64}
    log_post::Dict # Had to do dict since might want different output when (Iᵐ, γ) updated jointly
    dist::Metric
    I_prior::SPF{T}
    γ_prior::ContinuousUnivariateDistribution
    data::Vector{Path{T}}
    performance_measures::Dict
end 

struct SpfPosteriorModeConditionalMcmcOutput{T<:Union{Int, String}}
    γ_fixed::Float64
    I_sample::Vector{Path{T}}
    dist::Metric
    I_prior::SPF{T}
    data::Vector{Path{T}}
    performance_measures::Dict
end 

struct SpfPosteriorDispersionConditionalMcmcOutput{T<:Union{Int, String}}
    I_fixed::Path{T}
    γ_sample::Vector{Float64}
    γ_prior::ContinuousUnivariateDistribution
    data::Vector{Path{T}}
    performance_measures::Dict
end 



function Base.show(io::IO, output::T) where {T<:SpfPosteriorMcmcOutput}
    title = "MCMC Sample for SPF Posterior"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 

function Base.show(io::IO, output::T) where {T<:SpfPosteriorModeConditionalMcmcOutput}
    title = "MCMC Sample for SPF Posterior (Mode Conditional)"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 

function Base.show(io::IO, output::T) where {T<:SpfPosteriorDispersionConditionalMcmcOutput}
    title = "MCMC Sample for SPF Posterior (Dispersion Conditional)"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 



struct SisPosteriorMcmcOutput
    S_sample::Vector{InteractionSequence{Int}}
    γ_sample::Vector{Float64}
    posterior::SisPosterior
    suff_stat_trace::Vector{Float64}
    performace_measures::Dict
end 

struct SisPosteriorModeConditionalMcmcOutput
    γ_fixed::Float64
    S_sample::Vector{Vector{Path{Int}}}
    posterior::SisPosterior
    suff_stat_trace::Vector{Float64}
    performance_measures::Dict
end 

struct SisPosteriorDispersionConditionalMcmcOutput
    S_fixed::Vector{Path{Int}}
    γ_sample::Vector{Float64}
    posterior::SisPosterior
    performance_measures::Dict
end 


# ==========
#    SIM 
# ==========

struct SimPosteriorMcmcOutput
    S_sample::Vector{InteractionSequence{Int}}
    γ_sample::Vector{Float64}
    posterior::SimPosterior
    suff_stat_trace::Vector{Float64}
    performace_measures::Dict
end 

struct SimPosteriorModeConditionalMcmcOutput
    γ_fixed::Float64
    S_sample::Vector{Vector{Path{Int}}}
    posterior::SimPosterior
    suff_stat_trace::Vector{Float64}
    performance_measures::Dict
end 

struct SimPosteriorDispersionConditionalMcmcOutput
    S_fixed::Vector{Path{Int}}
    γ_sample::Vector{Float64}
    γ_prior::ContinuousUnivariateDistribution
    data::Vector{Vector{Path{Int}}}
    performance_measures::Dict
end 


