using InvertedIndices, StatsBase, Printf, ProgressMeter

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
vertex_countmap(x::InteractionSequence{T}) where {T<:Union{Int,String}} = countmap(vcat(x...))
vertex_counts(x::InteractionSequence{Int}) = counts(vcat(x...))
vertex_counts(x::InteractionSequence{Int}, levels::UnitRange{<:Integer}) = counts(vcat(x...), levels)
vertex_counts(x::InteractionSequence{Int}, k::Int) = counts(vcat(x...), k)
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

function sample_frechet_mean(
    data::InteractionSequenceSample{T}, 
    d::Union{InteractionSeqDistance,InteractionSetDistance};
    show_progress::Bool=false
    ) where T <:Union{Int, String}
    if show_progress
        iter = Progress(length(data), 1)
    end 
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
        if show_progress
            next!(iter)
        end 
    end 
    return data[ind_best], ind_best
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

