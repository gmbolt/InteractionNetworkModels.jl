export SingleMissingPredictive, PosteriorPredictive

struct SingleMissingPredictive
    S::InteractionSequence{Int}
    ind::Tuple{Int,Int}
    p::Vector{Float64}
end 

function Base.show(io::IO, pred::SingleMissingPredictive)
    title = "Missing Entry Predictive Distribution"
    println(io, title)
    println(io, "-"^length(title))
    println(io, "Observation: $(pred.S)")
    println(io, "Missing entry: $(pred.ind)")
end 

struct PosteriorPredictive
    posterior::Union{SisPosteriorMcmcOutput, SimPosteriorMcmcOutput}
    model_type::DataType
    function PosteriorPredictive(p::Union{SisPosteriorMcmcOutput, SimPosteriorMcmcOutput})
        T = typeof(p.posterior.S_prior)
        new(p, T)
    end 
end 