export SnfMcmcInsertDelete, SnfMcmcFlip

struct SnfMcmcInsertDelete
    ν::Int
    edgestore::Vector{Int}
    desired_samples::Int
    burn_in::Int
    lag::Int
end 

struct SnfMcmcFlip 
    ν::Int 
    desired_samples::Int 
    burn_in::Int
    lag::Int 
end 