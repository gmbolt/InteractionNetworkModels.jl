export SnfMcmcOutput

struct SnfMcmcOutput
    model::SNF
    sample::Vector{Matrix{Int}}
    performance_measures::Dict 
end 

function Base.show(io::IO, output::T) where {T<:SnfMcmcOutput}
    title = "MCMC Sample for SNF Model"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    for (key, value) in output.performance_measures
        println(io, key, ": ", value)
    end 
end 