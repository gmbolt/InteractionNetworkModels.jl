using Distances, LinearAlgebra
export SNF, MultigraphSNF

struct SNF{T<:Union{Int,Bool}}
    mode::Matrix{T}
    γ::Real
    d::Metric
    directed::Bool
    function SNF(mode::Matrix{S}, γ::Real, d::Metric) where {S<:Union{Int,Bool}}
        new{S}(mode, γ, d, issymmetric(mode))
    end
end 

const MultigraphSNF = SNF{Int}


