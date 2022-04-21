using LinearAlgebra
export CER

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

# Sampling functions 

function draw_sample!(
    out::Vector{Matrix},
    model::CER
    )
    for A in out 
        copy!(A, model.mode)
        for c in eachcol(A)
            for i in eachindex(c)
                if rand() < model.α
                    c[i] = !c[i]
                end 
            end 
        end 

    end 
end 

function draw_sample_no_copy!(
    out::Vector{Matrix{Bool}},
    model::CER
    )
    for A in out 
        for c in eachcol(A)
            for i in eachindex(c)
                if rand() < model.α
                    c[i] = !c[i]
                end 
            end 
        end 
    end 
end 

function draw_sample(
    model::CER,
    n::Int
    )
    out = [copy(model.mode) for i in 1:n]
    draw_sample_no_copy!(out, model)
    return out
end 

