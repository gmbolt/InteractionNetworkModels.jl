using GraphRecipes

export get_cooccurrence_matrix, get_cooccurrence_prob_matrix, plot_cooccurrence_graph
export get_informed_proposal_matrix

function get_cooccurrence_matrix(posterior::Union{SisPosterior,SimPosterior})
    V = length(posterior.V)
    C = zeros(Int, V, V)
    for S in posterior.data 
        C_tmp = zeros(Int, V, V)
        for I in S
            tmp_counts = counts(I, 1:V)
            vals = unique(I)
            # @show vals
            for j in 1:length(vals)
                for i in 1:(j-1)
                    C_tmp[vals[i],vals[j]] += 1 
                    C_tmp[vals[j],vals[i]] += 1 
                end 
                if tmp_counts[vals[j]] > 1 # If vertex appears more than once add to it's diag
                    C_tmp[vals[j], vals[j]] += 1
                end 
            end
        end 
        C += Int.(C_tmp .> 0)
    end 
    d = Dict(i => i for i in 1:V)
    return C, d, d
end 

# function get_cooccurrence_matrix(posterior::Union{SisPosterior{String},SimPosterior{String}})
    
#     vertex_set = posterior.V
#     V = length(vertex_set)
#     C = zeros(Int, V, V)
#     ind_map = Dict{String, Int}(v => i for (i,v) in enumerate(vertex_set))
#     for S in posterior.data 
#         C_tmp = zeros(Int, V, V)
#         for I in S
#             tmp_counts = countmap(I)
#             vals = unique(I)
#             for j in 1:length(vals)
#                 for i in 1:(j-1)
#                     C_tmp[ind_map[vals[i]],ind_map[vals[j]]] += 1 # only filling upper tri
#                     C_tmp[ind_map[vals[j]],ind_map[vals[i]]] += 1 # only filling upper tri
#                 end 
#                 if tmp_counts[vals[j]] > 1 # If vertex appears more than once add to it's diag
#                     C_tmp[ind_map[vals[j]], ind_map[vals[j]]] += 1
#                 end 
#             end
#         end 
#         C += Int.(C_tmp .> 0)
#     end 
#     ind_map_inv = Dict(val => key for (val, key) in zip(values(ind_map), keys(ind_map)))
#     return C, ind_map, ind_map_inv
# end

function get_cooccurrence_prob_matrix(C::Matrix{Int}, α::Real)
    
    @assert issymmetric(C) "Input matrix should be symmetric."

    V = size(C)[1]
    P = convert(Array{Float64,2}, C)

    for i in 1:V
        Z = sum(P[:,i])
        if  Z == 0.0
            P[:,i] = fill(1/V, V)
        else 
            P[:,i] = ((P[:,i] ./ Z) .+ α) ./ (1 + V*α)
        end 
    end 
    return CondProbMatrix(P)
end 

function get_informed_proposal_matrix(posterior::Union{SisPosterior,SimPosterior}, α::Real)
    C, vmap, vmap_inv = get_cooccurrence_matrix(posterior)
    P = get_cooccurrence_prob_matrix(C, α)
    return cumsum(P), vmap, vmap_inv
end 

function plot_cooccurrence_graph(posterior::Union{SisPosterior,SimPosterior})
    C, vmap, vmap_inv = get_cooccurrence_matrix(posterior)
    C₀ = copy(C)
    C₀[diagind(C₀)] .= 0
    fig = graphplot(
        C₀, names=vmap, 
        edge_width= (s,d,w) -> 2*C[s,d]/maximum(C),
        markersize=0.15, shape=:circle,
        fontsize=10,
        msw=0
    )
    display(fig)
end 
