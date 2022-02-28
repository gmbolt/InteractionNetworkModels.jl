using LinearAlgebra
export CER 

struct CER{T<:Union{Int,Bool}} 
    mode::Matrix{T}
    α::Real
    directed::Bool
end 

function CER(
    mode::Matrix{Int}, 
    α::Float64; 
    directed::Bool=issymmetric(mode)
    )
    @assert 0 < α < 1 "α must be within interval (0,1)."

    # If intger matrix given with just 0 and 1s make boolean 
    if prod(x->x∈[0,1], mode) 
        return CER(convert.(Bool,mode), α, directed)
    else 
        return CER(mode, α, directed)
    end 
end 


