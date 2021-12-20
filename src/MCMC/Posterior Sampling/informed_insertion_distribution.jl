using Distributions
export get_informed_insertion_dist

function get_informed_insertion_dist(
    posterior::Union{SisPosterior,SimPosterior},
    α::Float64
    )
    data = posterior.data 
    V = posterior.V
    p = zeros(length(V))
    for S in data   
        p .+= [i ∈ vcat(S...) for i in V]
    end 
    p ./= sum(p)
    p .+= α
    p ./= sum(p)
    return Categorical(p)
end 