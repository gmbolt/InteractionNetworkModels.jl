export SisMcmcOutput, SimMcmcOutput, SpfMcmcOutput


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

struct SpfMcmcOutput{T<:Union{String,Int}}
    model::SPF{T} # The model from which the sample was drawn
    sample::Vector{Path{T}}  # The sample
    a::Real # Acceptance Probability
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

function Base.show(io::IO, output::SpfMcmcOutput) 
    title = "MCMC Sample for Spherical Path Family (SPF)"
    println(io, title)
    println(io, "-"^length(title))
    println(io, "\nAcceptance probability: $(output.a)")
end 
