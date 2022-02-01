using LinearAlgebra

# Models 
# * Centered Erdos-Renyi (CER) model 
# * Spherical Network Family (SNF) model 

struct CER
    mode::Matrix{Int}
    α::Real
    directed::Bool
end 

function CER(
    mode::Matrix{Int}, 
    α::Float64; 
    directed::Bool=issymmetric(mode)
    )
    @assert 0 < α < 1 "α must be within interval (0,1)"
    CER(mode, α, directed)
end 

struct SNF 
    mode::Matrix{Int}
    γ::Real
    d::GraphDistance
    directed::Bool
end 

function SNF(
    mode::Matrix{Int}, 
    γ::Real, 
    d::GraphDistance;
    directed::Bool=issymmetric(mode)
    )
    @assert γ > 0 "γ must be positive."
    return SNF(mode, γ, d, directed)
end 

