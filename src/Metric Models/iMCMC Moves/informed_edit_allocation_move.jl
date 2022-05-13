
export InformedEditAllocationMove

struct InformedEditAllocationMove <: InvMcmcMove 
    ν::Int 
    P::Matrix{Float64}   # 
    counts::Vector{Int}  # Track acceptance 
    function InformedEditAllocationMove(;ν::Int=3)
        new(
            ν, 
            # zeros(Int,ν),
            # [0],
            [0,0]
        )
    end 
end 

Base.show(io::IO, x::EditAllocationMove) = print(io, "EditAllocationMove(ν=$(x.ν))")
