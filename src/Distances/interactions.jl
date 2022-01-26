using Distances, InvertedIndices

export InteractionDistance, PathDistance, LCS, FastLCS, NormLCS, FastNormLCS, lcs, get_lcs, lcs_norm, get_lcs_locations
export LSP, FastLSP
## Interaction Distances

abstract type InteractionDistance <: Metric end
abstract type PathDistance <: InteractionDistance end

struct LCS <: PathDistance end
struct NormLCS <: PathDistance end
struct FastLCS <: PathDistance 
    curr_row::Vector{Int}
    prev_row::Vector{Int}
    function FastLCS(K::Int)
        new(zeros(Int,K), zeros(Int,K))
    end 
end 

function Base.show(io::IO, d::FastLCS)
    println(io, "LCS (max interaction len. $(length(d.curr_row)))")
end 
struct FastNormLCS <: PathDistance
    curr_row::Vector{Int}
    prev_row::Vector{Int}
    function FastNormLCS(K::Int)
        new(zeros(Int,K), zeros(Int,K))
    end 
end 


# LCS
function (dist::LCS)(X::AbstractVector,Y::AbstractVector)::Float64

    n = length(X)
    m = length(Y)

    # @assert (n>0) & (m>0) "both paths must be either of type Nothing or of nonzero length."

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2
    prevRow = 0:m
    currRow = zeros(Int, m + 1)
    @inbounds begin 
        for i = 1:n
            currRow[1] = i
            for j = 1:m
                if X[i] == Y[j]
                    currRow[j+1] = prevRow[j]
                else
                    currRow[j+1] = min(currRow[j], prevRow[j+1]) + 1
                end
            end
            prevRow = copy(currRow)
        end 
    end 
    return currRow[end]
end

# With storage 
function (d::FastLCS)(
    X::AbstractVector,Y::AbstractVector
    )::Float64

    n = length(X)
    m = length(Y)

    @assert (n>0) & (m>0) "both paths must be either of type Nothing or of nonzero length."

    prev_row = view(d.prev_row, 1:(m+1))
    curr_row = view(d.curr_row, 1:(m+1))

    copy!(prev_row, 0:m)
    curr_row .= 0


    @inbounds begin 
        for i = 1:n
            curr_row[1] = i
            for j = 1:m
                if X[i] == Y[j]
                    curr_row[j+1] = prev_row[j]
                else
                    curr_row[j+1] = min(curr_row[j], prev_row[j+1]) + 1
                end
            end
            copy!(prev_row, curr_row)
        end
    end 
    return curr_row[m+1]
end

# Distances to the null 
function (dist::Union{LCS,FastLCS})(X::Nothing, Y::Path{T})::Float64 where {T<:Union{Int,String}}
    return length(Y)
end 
function (dist::Union{LCS,FastLCS})(X::Path{T}, Y::Nothing)::Float64 where {T<:Union{Int,String}}
    return length(X)
end
function (dist::Union{LCS,FastLCS})(X::Nothing, Y::Nothing)::Float64 where {T<:Union{Int,String}}
    return 0.0
end 



# Normalised LCS
function (dist::NormLCS)(X::AbstractVector,Y::AbstractVector)
    n = length(X)
    m = length(Y)
    d_lcs = lcs(X, Y)
    return 2 * d_lcs / (n + m + d_lcs)
end

function (d::FastNormLCS)(
    X::AbstractVector,Y::AbstractVector
    )::Float64

    n = length(X)
    m = length(Y)

    @assert (n>0) & (m>0) "both paths must be either of type Nothing or of nonzero length."

    @inbounds prev_row = view(d.prev_row, 1:(m+1))
    @inbounds curr_row = view(d.curr_row, 1:(m+1))

    copy!(prev_row, 0:m)
    curr_row .= 0

    # @show prev_row, curr_row

    @inbounds begin 
        for i = 1:n
            curr_row[1] = i
            for j = 1:m
                if X[i] == Y[j]
                    curr_row[j+1] = prev_row[j]
                else
                    curr_row[j+1] = min(curr_row[j], prev_row[j+1]) + 1
                end
            end
            copy!(prev_row, curr_row)
        end
        d_lcs = curr_row[m+1]
    end 
    return 2 * d_lcs / (n + m + d_lcs)
end

function (dist::Union{NormLCS,FastNormLCS})(X::Nothing, Y::Path{T}) where {T<:Union{Int,String}}
    return 1.0
end 
function (dist::Union{NormLCS,FastNormLCS})(X::Path{T}, Y::Nothing) where {T<:Union{Int,String}}
    return 1.0
end 
function (dist::Union{NormLCS,FastNormLCS})(X::Nothing, Y::Nothing) where {T<:Union{Int,String}}
    return 0.0
end 


# Get locations of longest common subseq (for visuals)

function get_lcs_locations(X::AbstractVector, Y::AbstractVector)
    
    C = zeros(Int, length(X)+1, length(Y)+1)

    C[:,1] = [i for i = 0:length(X)]
    C[1,:] = [i for i = 0:length(Y)]

    for j = 1:length(Y)
        for i = 1:length(X)
            C[i+1,j+1] = minimum([
            C[i,j] + 2*(X[i] != Y[j]),
            C[i,j+1] + 1,
            C[i+1,j] + 1
            ])
        end 
    end 

    i = length(X)+1; j = length(Y)+1;
    indx = Vector{Int}()
    indy = Vector{Int}() 
    # outputs = Vector{String}()
    while (i ≠ 1) & (j ≠ 1)
        if C[i,j] == (C[i-1,j] + 1)
            # println("Insert $(X[i-1]) of X")
            i = i-1
        elseif C[i,j] == (C[i, j-1] + 1)
            # println("Insert $(Y[j-1]) of Y")
            j = j-1 
        elseif C[i,j] == (C[i-1, j-1])
            # println("Sub $(X[i-1]) for $(Y[j-1])")
            pushfirst!(indx, i-1)
            pushfirst!(indy, j-1)
            i = i-1; j = j-1
        end
    end

    return indx, indy
end

# Longest Common Subpath (LSP)

struct LSP <: PathDistance end

function (dist::LSP)(X::AbstractVector,Y::AbstractVector)::Float64

    # Here we consider a Dynamic programming approach
    n = length(X)
    m = length(Y)

    @assert (n>0) & (m>0) "both paths must be either of type Nothing or of nonzero length."

    prev_row = zeros(Float64, m + 1)
    curr_row = zeros(Float64, m + 1)
    z = 0.0

    @inbounds begin 
        for i = 1:n
            for j = 1:m
                if X[i] == Y[j]
                    curr_row[j+1] = prev_row[j] + 1.0 # Subpath length increment
                    if curr_row[j+1] > z 
                        z = curr_row[j+1]
                    end 
                else
                    curr_row[j+1] = 0.0
                end
            end
            copy!(prev_row, curr_row)
        end
    end 
    return n + m - 2*z
end

struct FastLSP <: PathDistance 
    curr_row::Vector{Float64}
    prev_row::Vector{Float64}
    function FastLSP(K::Int)
        new(zeros(Float64,K), zeros(Float64,K))
    end 
end 

function (dist::FastLSP)(X::AbstractVector,Y::AbstractVector)::Float64
    # Here we take a Dynamic programming approach, but use pre-allocated arrays for storage.
    n = length(X)
    m = length(Y)

    @assert (n>0) & (m>0) "both paths must be either of type Nothing or of nonzero length."

    prev_row = view(dist.prev_row, 1:(m+1))
    curr_row = view(dist.curr_row, 1:(m+1))
    
    prev_row .= 0.0
    curr_row .= 0.0
    z = 0.0

    @inbounds begin 
        for i = 1:n
            for j = 1:m
                if X[i] == Y[j]
                    curr_row[j+1] = prev_row[j] + 1.0 # Subpath length increment
                    if curr_row[j+1] > z 
                        z = curr_row[j+1]
                    end 
                else
                    curr_row[j+1] = 0.0
                end
            end
            copy!(prev_row, curr_row)
        end
    end
    return n + m - 2*z
end

function (dist::Union{LSP,FastLSP})(X::Nothing, Y::Path{T})::Float64 where {T<:Union{Int,String}}
    return length(Y)
end 
function (dist::Union{LSP,FastLSP})(X::Path{T}, Y::Nothing)::Float64 where {T<:Union{Int,String}}
    return length(X)
end 
function (dist::Union{LSP,FastLSP})(X::Nothing, Y::Nothing) where {T<:Union{Int,String}}
    return 0.0
end 