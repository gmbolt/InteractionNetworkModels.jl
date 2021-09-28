using Distances, InvertedIndices

export InteractionDistance, PathDistance, LCS, FastLCS, NormLCS, FastNormLCS, lcs, get_lcs, lcs_norm, ACS, get_lcs_locations
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
struct InflatedLCS <: PathDistance end 
struct ACS <: PathDistance end 


# LCS
function (dist::LCS)(X::AbstractVector,Y::AbstractVector)::Float64

    n = length(X)
    m = length(Y)

    # If one or more is empty string return 0 or length of nonzero
    if (n == 0) || (m == 0)
        return n + m
    end

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2
    prevRow = 0:m
    currRow = zeros(Int, m + 1)
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
    return currRow[end]
end

# LCS with storage (save allocations)

function (dist::LCS)(
    X::AbstractVector,Y::AbstractVector,
    curr_row::Vector{Int}, prev_row::Vector{Int}
    )::Float64

    n = length(X)
    m = length(Y)

    # If one or more is empty string return 0 or length of nonzero
    if (n == 0) || (m == 0)
        return n + m
    end

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2


    @views prev_row[1:(m+1)] = 0:m
    @views curr_row[1:(m+1)] .= 0

    # @show prev_row, curr_row

    @views for i = 1:n
        curr_row[1] = i
        for j = 1:m
            if X[i] == Y[j]
                curr_row[j+1] = prev_row[j]
            else
                curr_row[j+1] = min(curr_row[j], prev_row[j+1]) + 1
            end
        end
        copy!(prev_row,curr_row)
    end
    return curr_row[m+1]
end

# With storage (as a field of the metric type)
function (d::FastLCS)(
    X::AbstractVector,Y::AbstractVector
    )::Float64

    n = length(X)
    m = length(Y)

    # If one or more is empty string return 0 or length of nonzero
    if (n == 0) || (m == 0)
        return n + m
    end

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2

    prev_row = view(d.prev_row, 1:(m+1))
    curr_row = view(d.curr_row, 1:(m+1))

    copy!(prev_row, 0:m)
    curr_row .= 0

    # @show prev_row, curr_row

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
    return curr_row[m+1]
end

function (d::FastNormLCS)(
    X::AbstractVector,Y::AbstractVector
    )::Float64

    n = length(X)
    m = length(Y)

    # If one or more is empty string return 0 or length of nonzero
    if (n == 0) || (m == 0)
        return n + m
    end

    # Code only needs previous row to update next row using the rule from
    # Wagner-Fisher algorithm

    #            Y
    #       0 1 2   ...
    #    X  1 0
    #       2

    prev_row = view(d.prev_row, 1:(m+1))
    curr_row = view(d.curr_row, 1:(m+1))

    copy!(prev_row, 0:m)
    curr_row .= 0

    # @show prev_row, curr_row

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
    return 2 * d_lcs / (n + m + d_lcs)
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



# Normalised LCS
function (dist::NormLCS)(X::AbstractVector,Y::AbstractVector)
    n = length(X)
    m = length(Y)
    if (n == 0) & (m == 0)
        return 0
    end
    d_lcs = lcs(X, Y)
    return 2 * d_lcs / (n + m + d_lcs)
end




lcs(X::AbstractVector,Y::AbstractVector) = LCS()(X,Y)
lcs_norm(X::AbstractVector, Y::AbstractVector) = NormLCS()(X,Y)

get_lcs(X::AbstractVector, Y::AbstractVector)::Int =  (length(X) + length(Y) - LCS()(X,Y))/2


# ACS 
function (dist::ACS)(X::AbstractVector, Y::AbstractVector)
    # Dynamic programming approach
    if length(X) < length(Y)  # Ensures first seq is longest
        ACS()(Y,X)
    else
        prev_row = fill(1, length(Y)+1)
        curr_row = fill(1, length(Y)+1)
        for i = 1:length(X)
            for j = 1:length(Y)
                if X[i] == Y[j]
                    # v = prev_row[j] + log(2
                    # println("match")
                    curr_row[j+1] = prev_row[j] * 2
                else
                    # println("no match")
                    curr_row[j+1] = prev_row[j+1] + curr_row[j] - prev_row[j]
                end
            end
            prev_row = copy(curr_row)
        end
        return 2^length(X) + 2^length(Y) - 2*curr_row[end]
    end
end


