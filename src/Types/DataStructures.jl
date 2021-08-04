using InvertedIndices, AutoHashEquals, StatsBase, Printf

export CondProbMatrix, CumCondProbMatrix, vertices, vertex_countmap, vertex_counts
export sample_frechet_mean, BoundedInteractionSequence
export insert_entry!, delete_entry!, delete_entries!, insert_interaction!, delete_interaction!
export insert_interaction_rand!, delete_insert_interaction!
export copy_interaction!
# abstract type Interaction end

# =============================================
# Finding unique vertices and their frequencies
# =============================================

# This included implementations for paths, vectors of paths, vectors of vectors of paths

vertices(p::Path) = unique(p)
vertices(x::InteractionSequence) = unique(vcat(x...))

# Helper functions
vertex_countmap(x::Path{T}) where {T<:Union{Int,String}} = StatsBase.counts(x)
vertex_counts(x::Path{T}) where {T<:Union{Int,String}} = StatsBase.counts(x)
vertex_countmap(x::InteractionSequence{T}) where {T<:Union{Int,String}} = countmap(vertices(x))
vertex_counts(x::InteractionSequence{Int}) = counts(vertices(x))
vertex_counts(x::InteractionSequence{Int}, levels::UnitRange{<:Integer}) = counts(vertices(x), levels)
vertex_counts(x::InteractionSequence{Int}, k::Int) = counts(vertices(x), k)
function vertex_counts(x::Union{InteractionSequence{T}, Path{T}}, ref::Vector{T}) where {T<:Union{Int,String}}
    d = vertex_countmap(x)
    return [ref[i] in keys(d) ? d[ref[i]] : 0 for i in 1:length(ref)]
end 


function Base.vec(data::InteractionSequenceSample)
    return vcat(vcat(data...)...)
end

# =======================================================
# Sample Frechet Means
# =======================================================

function sample_frechet_mean(data::InteractionSequenceSample{T}, d::PathDistance) where T <:Union{Int, String}
    z_best = Inf
    ind_best = 1
    n = length(data)
    for i in 1:n
        z = 0.0
        j = 1 + (i==1)
        while (z < z_best) & (j ≤ n)
            z += d(data[i], data[j])
            j += 1 + (j==(i-1))
        end 
        if z < z_best 
            z_best = copy(z)
            ind_best = i
        end 
    end 
    return data[ind_best], ind_best, z_best
end 

# Probability Matrices - Used in informed proposal for SIS/SIM posterior sampling

struct CondProbMatrix
    data::Matrix{Float64}
    function CondProbMatrix(data::Matrix{Float64})
        @assert prod(sum(data, dims=1) .≈ 1.0) "Input not valid. Must have column sums equal to one."
        new(data)
    end
end 

struct CumCondProbMatrix 
    data::Matrix{Float64}
    function CumCondProbMatrix(data::Matrix{Float64})
        @assert size(data)[1] == (size(data)[2]+1) "Incorrect dimensions. Must have columns as cumulative probabilities, with 0 first entry."
        @assert prod(data[1,:] .== 0.0) "Incorrect initial entries. First entry of each column must be 0.0" 
        @assert prod(data[end,:] .≈ 1.0) "Incorrect final entries. Final entry of each column must be 1.0"  
        new(data)
    end 
end     

Base.size(P::CondProbMatrix) = size(P.data)
Base.getindex(P::CondProbMatrix, ind1, ind2) = P.data[ind1, ind2]
# Base.show(io::IO, P::CondProbMatrix) = print(io, P.data)

function Base.show(io::IO, P::CondProbMatrix)
    (n,m) = size(P)
    println(io, "$(n)x$(m) Conditional Probability Matrix (Colwise)")
    for i in 1:n
        for j in 1:m
            print(io, @sprintf("%.2f ", P[i,j]))
        end 
        print(io, "\n")
    end 
end 

Base.size(P::CumCondProbMatrix) = size(P.data)
Base.getindex(P::CumCondProbMatrix, ind1, ind2) = P.data[ind1, ind2]
function Base.show(io::IO, P::CumCondProbMatrix)
    (n,m) = size(P)
    println(io, "$(n)x$(m) Cumulative Conditional Probability Matrix (Colwise)")
    for i in 1:n
        for j in 1:m
            print(io, @sprintf("%.2f ", P[i,j]))
        end 
        print(io, "\n")
    end 
end
function CumCondProbMatrix(P::CondProbMatrix)
    P_cusum = cumsum(P.data, dims=1)
    P_cusum = [zeros(1, size(P)[1]); P_cusum]
    return CumCondProbMatrix(P_cusum)
end 

function Base.cumsum(P::CondProbMatrix)
    P_cusum = cumsum(P.data, dims=1)
    P_cusum = [zeros(1, size(P)[1]); P_cusum]
    return CumCondProbMatrix(P_cusum)
end 

# =======================
# Bounded Interaction Sequences 

struct BoundedInteractionSequence{T<:Union{Int,String}} 
    data::Matrix{T}
    dims::Vector{Int}
    K_inner::Int
    K_outer::Int
    function BoundedInteractionSequence(data::Matrix{S}) where {S<:Union{Int, String}}
        K_inner, K_outer = (size(data).-1)
        dims = [(findfirst(iszero, x) - 1) for x in eachcol(data)]
        push!(dims, findfirst(iszero, dims)-1)
        # @show dims
        # @show K_inner, K_outer
        new{S}(data, dims, K_inner, K_outer)
    end 
end 

function BoundedInteractionSequence(x::InteractionSequence, K_inner::Int, K_outer::Int)
    data = zeros(Int, K_inner+1, K_outer+1)
    for (i,path) in enumerate(x)
        m = length(path)
        @views copy!(data[1:m,i], path)
    end 
    # @show data
    BoundedInteractionSequence(data)
end 

BoundedInteractionSequence(x::InteractionSequence) = BoundedInteractionSequence(x, maximum(length.(x)), length(x))



Base.getindex(x::BoundedInteractionSequence, i::Int, j::Int) = x.data[i, j]
Base.getindex(x::BoundedInteractionSequence, i::Int,) = view(x.data, 1:x.dims[i], i)
Base.setindex!(x::BoundedInteractionSequence, i::Int, j::Int, val::Int) = setindex!(x.data, i, j, val)
Base.length(x::BoundedInteractionSequence) = x.dims[end]
Base.size(x::BoundedInteractionSequence) = (length(x),)

function Base.show(
    io::IO, 
    ::MIME"text/plain", 
    x::BoundedInteractionSequence{T}) where {T<:Union{Int,String}} 
    println(io, "BoundedInteractionSequence{$(T)}")
    println(io, "K_inner = $(x.K_inner); K_outer = $(x.K_outer)")
    for i in 1:length(x)
        println(io, x[i])
    end 
end 

function insert_entry!(
    x::BoundedInteractionSequence,
    i::Int, 
    j::Int,
    item::Int
    )
    @assert j ≤ (x.dims[i]+1) "Trying to insert beyond inner path length+1."
    for k in (x.dims[i]+1):-1:(j+1)
        x[k,i] = x[k-1,i]
    end 
    x[j,i] = item
    x.dims[i] += 1
end  

function delete_entry!(
    x::BoundedInteractionSequence,
    i::Int, 
    j::Int;
    null::Int=0
    )
    @assert x.dims[i] > 1 "Cannot delete from length 1 interaction."
    for k in j:(x.dims[i]-1)
        x[k,i] = x[k+1,i]
    end 
    x[x.dims[i],i] = null
    x.dims[i] -= 1
end 


function insert_interaction!(
    x::BoundedInteractionSequence, 
    i::Int, 
    item::Vector{Int}
    )
    # Shift entries
    @assert length(x) < x.K_outer "Invalid insertion attempt. At max num. of interactions and trying to insert."
    for j in (length(x)+1):-1:(i+1)
        for k in 1:(max(x.dims[j], x.dims[j-1])+1)
            x[k,j] = x[k, j-1]
        end 
        x.dims[j] = x.dims[j-1]
    end 
    for k in 1:length(item)
        x[k,i] = item[k]
    end 
    for k in (length(item)+1):x.dims[i]
        x[k,i] = 0
    end 

    # Update dims
    x.dims[i] = length(item) # new entry 
    x.dims[end] += 1
end 

function insert_interaction!(
    x::BoundedInteractionSequence, 
    i::Int, 
    item::Int
    )
    @assert length(x) < x.K_outer "Invalid insertion attempt. At max num. of interactions and trying to insert."
    # Shift entries
    for j in (length(x)+1):-1:(i+1)
        for k in 1:max(x.dims[j], x.dims[j-1])
            x[k,j] = x[k, j-1]
        end 
        x.dims[j] = x.dims[j-1]
    end 
    x[i,1] = item 
    for k in 2:x.dims[i]
        x[k,i] = 0
    end 
    x.dims[i] = 1
    x.dims[end] += 1
end 

function delete_interaction!(
    x::BoundedInteractionSequence, 
    i::Int
    )
    @assert length(x) > 1 "Invalid deletion attempt. Cannot delete from length one interaction sequence."
    for j in i:length(x)
        for k in 1:max(x.dims[j], x.dims[j+1])
            x[k, j] = x[k, j+1]
        end 
        x.dims[j] = x.dims[j+1]
    end 
    x.dims[end] -= 1 # Update length of x
end 


function insert_interaction_rand!(
    d::PathPseudoUniform, 
    x::BoundedInteractionSequence, 
    i::Int
    )
    @assert length(x) < x.K_outer "Invalid insertion attempt. At max num. of interactions and trying to insert."

    m = rand(d.length_dist)

    @assert m ≤ x.K_inner "Invalide insertion attempt. Interaction to be inserted is longer than K_inner"
    
    insert_interaction!(x, i, 1) # Insert dummy interaction [1]
    x.dims[i] = m # Indexing x[i] on next line returns a view, but we must first set the length to m as desired
    @views sample!(d.vertex_set, x[i])
    x.dims[i]
end 



function delete_insert_interaction!(
    x::BoundedInteractionSequence, 
    i::Int, 
    δ::Int, d::Int, 
    ind_del::AbstractVector{Int},
    ind_add::AbstractVector{Int}, 
    vals::AbstractVector{Int}
    )

    @views for (k, index) in enumerate(ind_del[1:d])
        delete_entry!(x, i, index - k + 1)
    end 
    @views for (index, val) in zip(ind_add[1:(δ-d)], vals[1:(δ-d)])
        insert_entry!(x, i, index, val)
    end 

end 

function copy_interaction!(
    x::BoundedInteractionSequence, 
    y::BoundedInteractionSequence, 
    i::Int, j::Int
    )

    # Copy jth of y to ith of xs
    for k in 1:max(x.dims[i], y.dims[j])
        x[k,i] = y[k,j]
    end 
    x.dims[i] = y.dims[j]
end 

function Base.copy!(x::BoundedInteractionSequence, y::BoundedInteractionSequence)

    for i in 1:max(length(x), length(y))
        for k in 1:max(x.dims[i], y.dims[i])
            x[k,i] = y[k,i]
        end 
        x.dims[i] = y.dims[i]
    end 

end 

