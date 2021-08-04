export multigraph_edge_list

# The following is useful for when we want to plot a path, it takes in a path and outputs
# a Vecot
"""
`multigraph_edge_list(x::Path)`

This is a helper function which can be used in conjunction with `GraphRecipes.graphplot()` to visualise a path.

Output is tuple `(edge_list, label_map)` where 
* `edge_list::Vector{Vector{Int}}` describes all edges, whereby edge_list[1] stores all edges starting at node 1, edge_list[2] those starting at 2 and so on...
* `label_map::Dict` this maps the integer labels of nodes back to the of the input data, e.g. label_map[1] = "a" means node 1 represents the vertex "a" in the path `p`. 

To plot path with labels call

`graphplot(edge_list, names=label_map)`
"""
function multigraph_edge_list(p::Path{T}) where {T<:Union{Int, String}}
    edge_list = [Vector{Int}() for i in unique(p.data)]
    # We now loop over the vector and store the mapping from integers to strings
    ind = 1
    label_map = Dict{T, Int}(p[1]=>ind)
    ind+=1
    for i in 1:(length(p)-1)
        if ! haskey(label_map, p[i+1])
            label_map[p[i+1]] = ind
            ind += 1
        end 
        push!(edge_list[label_map[p[i]]], label_map[p[i+1]])
    end 
    # Invert the dictionary 
    label_map = Dict{Int, T}(val => key for (key, val) in label_map)
    return edge_list, label_map
end 



function multigraph_edge_list(x::Vector{Path{T}}) where {T<:Union{Int, String}}
    V = length(vertices(x))
    # @show V
    edge_lists = Vector{Vector{Vector{Int}}}()
    label_map = Dict{T, Int}()
    ind = 1
    for p in x
        tmp_edge_list = [Vector{Int}() for i in 1:V]
        if ! haskey(label_map, p[1])
            label_map[p[1]] = ind
            ind += 1
        end 
        # @show tmp_edge_list, p
        for i in 1:(length(p)-1)
            if ! haskey(label_map, p[i+1])
                label_map[p[i+1]] = ind
                ind += 1
            end 
            # @show label_map
            push!(tmp_edge_list[label_map[p[i]]], label_map[p[i+1]])
        end 
        # @show tmp_edge_list
        push!(edge_lists, tmp_edge_list)
    end 
    # Invert the dictionary 
    label_map = Dict{Int, T}(val => key for (key, val) in label_map)
    return edge_lists, label_map
end 

