export SnfMcmcInsertDelete, SnfMcmcFlip, SnfMcmcSysGibbs, SnfMcmcRandGibbs
export SnfMcmcSampler

abstract type SnfMcmcSampler end 

struct SnfMcmcInsertDelete <: SnfMcmcSampler
    ν::Int
    nz_edges::Vector{Int}
    desired_samples::Int
    burn_in::Int
    lag::Int
end 

function SnfMcmcInsertDelete(
    ;
    ν::Int=3, 
    desired_samples::Int=1000, 
    burn_in::Int=0, 
    lag::Int=1
    )
    return SnfMcmcInsertDelete(ν, Int[], desired_samples, burn_in, lag)
end 

struct SnfMcmcFlip <: SnfMcmcSampler
    ν::Int 
    desired_samples::Int 
    burn_in::Int
    lag::Int 
end 

struct SnfMcmcSysGibbs <: SnfMcmcSampler
    ν::Int
    desired_samples::Int
    burn_in::Int
    lag::Int
end 

function SnfMcmcSysGibbs(
    ;
    ν::Int=3, 
    desired_samples::Int=1000, 
    burn_in::Int=0, 
    lag::Int=1
    )
    return SnfMcmcSysGibbs(ν, desired_samples, burn_in, lag)
end 

struct SnfMcmcRandGibbs <: SnfMcmcSampler
    ν::Int  # Neighbourhood size parameter for component updates 
    desired_samples::Int
    burn_in::Int
    lag::Int
end 

function SnfMcmcRandGibbs(
    ; 
    ν::Int=3, 
    desired_samples::Int=1000, 
    burn_in::Int=0, 
    lag::Int=1
    )
    return SnfMcmcRandGibbs(ν, desired_samples, burn_in, lag)
end 