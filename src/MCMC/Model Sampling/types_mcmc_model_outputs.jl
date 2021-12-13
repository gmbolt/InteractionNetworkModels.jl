export SisMcmcOutput, SimMcmcOutput, SpfMcmcOutput


struct SisMcmcOutput
    model::SIS # The model from which the sample was drawn
    sample::Vector{Vector{Path{Int}}}  # The sample
    performance_measures::Dict  # Dictionary of performance measures key => value, e.g. "acceptance probability" => 0.25
end 

struct SimMcmcOutput
    model::SIM
    sample::Vector{Vector{Path{Int}}}  # The sample
    performance_measures::Dict  # Dictionary of performance measures key => value, e.g. "acceptance probability" => 0.25
end 

struct SpfMcmcOutput{T<:Union{String,Int}}
    model::SPF{T} # The model from which the sample was drawn
    sample::Vector{Path{T}}  # The sample
    a::Real # Acceptance Probability
end 


function Base.show(io::IO, output::T) where {T<:SisMcmcOutput}
    title = "MCMC Sample for SIS Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 

function Base.show(io::IO, output::T) where {T<:SimMcmcOutput}
    title = "MCMC Sample for SIM Model"
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
