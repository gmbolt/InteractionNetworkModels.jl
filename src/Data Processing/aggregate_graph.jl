export path_to_edge_list, get_aggregate_graph

function path_to_edge_list(x::Vector{T}) where {T}
    out = Tuple{T,T}[]
    for i in 1:(length(x)-1)
        push!(out, (x[i], x[i+1]))
    end 
    return out 
end
function path_to_edge_list(x::Vector{String}; relabel=true)
    if relabel
        x_int, rev_map = string_to_int(x)
        return path_to_edge_list(x_int), rev_map
    else 
        out = Tuple{String,String}[]
        for i in 1:(length(x)-1)
            push!(out, (x[i], x[i+1]))
        end 
        return out 
    end 
end 

function get_aggregate_graph(S::InteractionSequence{Int})
    V = length(unique(vcat(S...)))
    A = zeros(Int, V, V)
    for path in S
        for i in 1:(length(path)-1)
            src, dest = (path[i],path[i+1])
            A[src,dest] += 1
        end 
    end 
    return A
end 

function get_aggregate_graph(S::InteractionSequence{Int}, V::Int)
    A = zeros(Int, V, V)
    for path in S
        for i in 1:(length(path)-1)
            src, dest = (path[i],path[i+1])
            A[src,dest] += 1
        end 
    end 
    return A
end 


function get_aggregate_graph(S::InteractionSequence{String})
    S_int, rev_map = string_to_int(S)
    return get_aggregate_graph(S_int), rev_map
end 