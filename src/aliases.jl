export Path, InteractionSequence, InteractionSequenceSample

const Path{T} = Vector{T} where {T<:Union{Int, String}} 
const InteractionSequence{T} = Vector{Vector{T}} where {T<:Union{Int, String}}
const InteractionSequenceSample{T} = Vector{Vector{Vector{T}}} where {T<:Union{Int, String}}

