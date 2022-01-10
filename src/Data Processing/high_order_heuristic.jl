using IterTools

export get_subseqs, get_subseqs!, get_subseq_counts!, get_subseq_counts
export get_subseq_proportions, dict_rank

function get_subseqs!(
    out::Set{Vector{T}},
    S::InteractionSequence{T}, 
    r::Int    
    ) where {T<:Union{Int,String}}
    for path in S 
        n = length(path)
        for ind in subsets(1:n, r)
            push!(out, path[ind])
        end 
    end 
end 

function get_subseqs(S::InteractionSequence{T}, r::Int) where {T<:Union{Int,String}}
    out = Set{Vector{T}}()
    get_subseqs!(out, S, r)
    return out
end     

function get_subseq_counts!(
    out::Dict{Vector{T},S}, 
    data::InteractionSequenceSample{T},
    r::Int
    ) where {T<:Union{Int,String},S<:Real}

    uniq_paths = Set{Vector{T}}()
    for x in data 
        empty!(uniq_paths)
        get_subseqs!(uniq_paths, x, r)
        for path in uniq_paths
            out[path] = get(out, path, 0) + 1
        end 
    end 
end 

function get_subseq_counts(data::InteractionSequenceSample{T}, r::Int) where {T<:Union{Int,String}}
    out = Dict{Vector{T},Int}()
    get_subseq_counts!(out, data, r)
    return out
end 

function get_subseq_proportions(data::InteractionSequenceSample{T}, r::Int) where {T<:Union{Int,String}}
    out = Dict{Vector{T},Float64}()
    n = length(data)
    get_subseq_counts!(out, data, r)
    for key in keys(out)
        out[key] /= n 
    end 
    return out
end 

function dict_rank(d::Dict{T, S}) where {T<:Any, S<:Real}
    pairs = collect(d) # Makes vector of pairs
    vals = collect(values(d))
    ind = sortperm(vals, rev=true)
    return pairs[ind]
end 