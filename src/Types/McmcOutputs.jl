using RecipesBase, StatsBase

export SpfPosteriorMcmcOutput, SpfPosteriorModeConditionalMcmcOutput, SpfPosteriorDispersionConditionalMcmcOutput
export SpfMcmcOutput
export print_map_est

# ==========================
#      SPF 
# ==========================


# From Model
struct SpfMcmcOutput{T<:Union{String,Int}}
    model::SPF{T} # The model from which the sample was drawn
    sample::Vector{Path{T}}  # The sample
    a::Real # Acceptance Probability
end 

function Base.show(io::IO, output::SpfMcmcOutput) 
    title = "MCMC Sample for Spherical Path Family (SPF)"
    println(io, title)
    println(io, "-"^length(title))
    println(io, "\nAcceptance probability: $(output.a)")
end 

# Posteriors 
struct SpfPosteriorMcmcOutput{T<:Union{Int, String}}
    I_sample::Vector{Path{T}}
    γ_sample::Vector{Float64}
    log_post::Dict # Had to do dict since might want different output when (Iᵐ, γ) updated jointly
    dist::PathDistance
    I_prior::SPF{T}
    γ_prior::ContinuousUnivariateDistribution
    data::Vector{Path{T}}
    performance_measures::Dict
end 

struct SpfPosteriorModeConditionalMcmcOutput{T<:Union{Int, String}}
    γ_fixed::Float64
    I_sample::Vector{Path{T}}
    dist::PathDistance
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



@recipe function f(output::T) where {T<:SpfPosteriorMcmcOutput}
    xguide --> "Index"
    yguide --> "γ"
    legend --> false
    size --> (800, 300)
    label := nothing
    output.γ_sample
end 


@recipe function f(output::SpfPosteriorMcmcOutput{T}, I_true::Path{T}) where T <:Union{Int, String}
    I_sample = output.I_sample
    xguide --> "Index"
    yguide --> ["Distance from Truth" "γ"]
    legend --> false
    size --> (800, 600)
    label := [nothing nothing]
    layout := (2,1)
    y1 = map(x -> output.dist(I_true, x), I_sample)
    y2 = output.γ_sample
    hcat(y1, y2)
end 

@recipe function f(output::SpfPosteriorModeConditionalMcmcOutput{T}, I_true::Path{T}) where T <:Union{Int, String}
    I_sample = output.I_sample
    xguide --> "Index"
    yguide --> "Distance from Truth"
    legend --> false
    size --> (800, 600)
    y1 = map(x -> output.dist(I_true, x), I_sample)
    y1
end 

@recipe function f(output::SpfPosteriorDispersionConditionalMcmcOutput{T}) where T <:Union{Int, String}
    xguide --> "Index"
    yguide --> "γ"
    legend --> false
    size --> (800, 600)
    output.γ_sample
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

function StatsBase.addcounts!(d::Dict{Path{T}, Real}, x::Path) where {T<:Union{Int,String}}
    d[x] = get!(d, x, 0) + 1
end 

function StatsBase.countmap(x::Vector{Path{T}}) where {T<:Union{Int,String}}
    d = Dict{Path{T}, Real}()
    for path in x 
        addcounts!(d, path)
    end 
    return d
end 


function print_map_est(output::T; top_num::Int=5) where {T<:Union{SpfPosteriorModeConditionalMcmcOutput, SpfPosteriorMcmcOutput}}
    d = Dict{Path, Real}()
    for x in output.I_sample
        addcounts!(d, x)
    end 
    for key in keys(d)
        d[key] /= length(output.I_sample)
    end 
    counts = sort(collect(d), by=x->x[2], rev=true)
    title = "\nPosterior probability of modal interaction Iᵐ"
    println(title)
    println("-"^length(title), "\n")
    for i in 1:min(top_num, length(d))
        println(counts[i][2],"  ", counts[i][1])
    end    
    println("\n...showing top $(min(top_num, length(d))) interactions.")
end 

# ==========================
#      SIS/SIM
# ==========================

struct SisMcmcOutput{T<:Union{Int, String}}
    model::SIS{T} # The model from which the sample was drawn
    sample::Vector{Vector{Path{T}}}  # The sample
    performance_measures::Dict  # Dictionary of performance measures key => value, e.g. "acceptance probability" => 0.25
end 

struct SimMcmcOutput{T<:Union{Int, String}}
    model::SIM{T}
    sample::Vector{Vector{Path{T}}}  # The sample
    performance_measures::Dict  # Dictionary of performance measures key => value, e.g. "acceptance probability" => 0.25
end 

function Base.show(io::IO, output::T) where {T<:SisMcmcOutput}
    title = "MCMC Sample for Spherical Interaction Sequence (SIS) Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 

function Base.show(io::IO, output::T) where {T<:SimMcmcOutput}
    title = "MCMC Sample for Spherical Interaction Multiset (SIM) Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 