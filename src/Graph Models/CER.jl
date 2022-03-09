using LinearAlgebra
export CER, CerPosterior

struct CER 
    mode::Matrix{Bool}
    α::Real
    directed::Bool
    self_loops::Bool
end 

function CER(
    mode::Matrix{Int}, 
    α::Float64; 
    directed::Bool=!issymmetric(mode),
    self_loops::Bool=any(diag(mode).>0)
    )
    @assert 0 < α < 1 "α must be within interval (0,1)."

    # If intger matrix given with just 0 and 1s make boolean 
    if prod(x->x∈[0,1], mode) 
        return CER(convert.(Bool,mode), α, directed, self_loops)
    else 
        error("Entries must be boolean or 0/1 integers.")
    end 
end 


function CER(
    mode::Matrix{Bool}, 
    α::Float64; 
    directed::Bool=!issymmetric(mode),
    self_loops::Bool=any(diag(mode))
    )
    @assert 0 < α < 1 "α must be within interval (0,1)."

    # If intger matrix given with just 0 and 1s make boolean 
    return CER(mode, α, directed, self_loops)
end 


# Poster

struct CerPosterior
    data::Vector{Matrix{Bool}}
    G_prior::CER 
    α_prior::UnivariateDistribution
    sample_size::Int
    function CerPosterior(
        data::Vector{Matrix{Bool}},
        G_prior::CER, 
        α_prior::UnivariateDistribution
        ) 

        new(data,G_prior,α_prior,length(data))
    end
end 

function CerPosterior(
    data::Vector{Matrix{Int}},
    G_prior::CER, 
    α_prior::UnivariateDistribution
    ) 

    if prod(y->prod(x->x∈[0,1], y), data)
        data_bool = map(x->convert(Matrix{Bool},x), data)
        CerPosterior(data_bool, G_prior, α_prior)
    else 
        error("Entries must be boolean or 0/1 integers.")
    end 

end 