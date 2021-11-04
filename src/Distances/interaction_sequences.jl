using StatsBase, Distances

export InteractionSeqDistance
export GED, FastGED, FpGED, NormFpGED, AvgSizeFpGED

abstract type InteractionSeqDistance <: Metric end

struct GED{T<:InteractionDistance} <:InteractionSeqDistance
    ground_dist::T
end

struct FastGED{T<:InteractionDistance} <:InteractionSeqDistance
    ground_dist::T
    curr_row::Vector{Float64}
    prev_row::Vector{Float64}
    function FastGED(ground_dist::S, K::Int) where {S<:InteractionDistance}
        new{S}(ground_dist, zeros(Float64, K), zeros(Float64, K))
    end 
end
function Base.show(io::IO, d::FastGED)
    print(io, "GED (max num. interactions $(length(d.curr_row))) with $(d.ground_dist) ground distance.")
end 

struct FpGED{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real
end

struct AvgSizeFpGED{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real
end 
struct NormFpGED{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real # Penalty
end



# GEDs

function (d::GED)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        d₀ = d.ground_dist
        prev_row = pushfirst!(cumsum([d₀(T[], p) for p in S2]), 0.0);
        curr_row = zeros(Float64, length(S2) + 1);

        for i = 1:length(S1)
            curr_row[1] = prev_row[1] + length(S1[i])
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                curr_row[j+1] = min(prev_row[j] + d₀(S1[i], S2[j]),
                                        prev_row[j+1] + d₀(T[], S1[i]),
                                        curr_row[j] + d₀(T[], S2[j]))
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[end]
    end
end


function (d::FastGED)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        d₀ = d.ground_dist
        prev_row = view(d.prev_row, 1:(length(S2)+1))
        curr_row = view(d.curr_row, 1:(length(S2)+1))
        # prev_row = d.prev_row
        # curr_row = d.curr_row
        Λ = T[]

        prev_row[1] = 0.0
        for i in 1:length(S2)
            prev_row[i+1] = prev_row[i] + d₀(Λ, S2[i])
        end 
        curr_row .= 0.0
        

        @views for i = 1:length(S1)
            curr_row[1] = prev_row[1] + length(S1[i])
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                curr_row[j+1] = min(
                    prev_row[j] + d₀(S1[i], S2[j]), 
                    prev_row[j+1] + d₀(Λ, S1[i]), 
                    curr_row[j] + d₀(Λ, S2[j])
                    )
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[length(S2) + 1]
    end
end

# With substitution dictionary (pairwise distances, with d([],I) stored at end row/column

function (d::GED)(
    S1::Vector{Path{T}}, S2::Vector{Path{T}}, 
    D::Dict{Path{T},Dict{Path{T}, Float64}}
    ) where {T<:Union{Int,String}}


    prev_row = pushfirst!(cumsum([D[Path(T[])][p] for p in S2]), 0.0);
    curr_row = zeros(Float64, length(S2) + 1);
    # Julia indexes in column major order, so we index over 
    for i = 1:length(S1)
        curr_row[1] = prev_row[1] + length(S1[i])
        for j = 1:(length(S2))
            # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
            curr_row[j+1] = minimum([prev_row[j] + D[S1[i]][S2[j]], 
                                    prev_row[j+1] + D[S1[i]][Path(T[])],
                                    curr_row[j] + D[Path(T[])][S2[j]] ])
        end
        # @show curr_row
        prev_row = copy(curr_row)
    end
    return curr_row[end]
end


function (d::FpGED)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        prev_row = d.ρ/2 * collect(0:length(S2));
        curr_row = zeros(Float64, length(S2) + 1);

        for i = 1:length(S1)
            curr_row[1] = i*d.ρ
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                curr_row[j+1] = minimum([prev_row[j] + d.ground_dist(S1[i], S2[j]),
                                        prev_row[j+1] + d.ρ/2,
                                        curr_row[j] + d.ρ/2])
            end
            # @show curr_row
            prev_row = copy(curr_row)
        end
        return curr_row[end]
    end
end

function (d::AvgSizeFpGED)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

    d_ged = FpGED(d.ground_dist, d.ρ)(S1, S2)

    return d_ged + (mean(length.(S1)) - mean(length.(S2)))^2

end 

# Normed GED

function (d::NormFpGED)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)
        d(S2, S1)
    else
        tmp_d = FpGED(d.ground_dist, d.ρ)(S1, S2)
        return 2*tmp_d / ( d.ρ*(length(S1) + length(S2)) + tmp_d )
    end
end

# Get a GED Matching. This is useful for testing, i.e. seeing which pairs
# were matched explicitly.

function get_matching(d::GED, S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    
    d₀ = d.ground_dist
    # First find the substitution matrix
    C = zeros(Float64, length(S1)+1, length(S2)+1)
    C[:,1] = pushfirst!(cumsum([d₀(Path(), p) for p in S1]), 0.0);
    C[1,:] = pushfirst!(cumsum([d₀(Path(), p) for p in S2]), 0.0);

    for j in 1:length(S2)
        for i in 1:length(S1)
            C[i+1,j+1] = minimum([
                C[i,j] + d₀(S1[i], S2[j]),
                C[i,j+1] + d₀(Path(), S2[j]),
                C[i+1,j] + d₀(Path(), S1[i])
            ])
        end 
    end

    # Now retrace steps to determine an optimal matching
    i, j = size(C)
    outputs = Vector{String}()
    while (i ≠ 1) | (j ≠ 1)
        if C[i,j] == (C[i-1,j] + d₀(Path(), S1[i-1]) )
            pushfirst!(outputs, "$(S1[i-1]) --> Nothing")
            i = i-1
        elseif C[i,j] == (C[i,j-1] + d₀(Path(), S2[j-1]))
            pushfirst!(outputs, "Nothing --> $(S2[j-1])")
            j = j-1
        else
            pushfirst!(outputs, "$(S1[i-1]) ---> $(S2[j-1])")
            i = i-1; j = j-1
        end
    end
    # @show outputs
    title = "\nOptimal Matching Print-out for $d Distance"
    println(title)
    println("-"^length(title), "\n")
    println("The cheapest way to do the tranformation...\n")
    println(S1, "---->", S2)
    println("\n...is the following series of edits...\n")
    for statement in outputs
        println(statement)
    end
    return C

end 

function get_matching(d::FpGED, S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

    # First find the substitution matricx
    C = zeros(Float64, length(S1)+1, length(S2)+1)
    
    C[:,1] = [d.ρ/2 * i for i = 0:length(S1)]
    C[1,:] = [d.ρ/2 * i for i = 0:length(S2)]
    
    for j = 1:length(S2)
        for i = 1:length(S1)
            C[i+1,j+1] = minimum([
            C[i,j] + d.ground_dist(S1[i], S2[j]),
            C[i,j+1] + d.ρ/2,
            C[i+1,j] + d.ρ/2
            ])
        end
    end

    # Now retrace steps to determine an optimal matching
    i, j = size(C)
    outputs = Vector{String}()
    while (i ≠ 1) | (j ≠ 1)
        if C[i,j] == (C[i-1,j] + d.ρ/2 )
            pushfirst!(outputs, "$(S1[i-1]) --> Nothing")
            i = i-1
        elseif C[i,j] == (C[i,j-1] + d.ρ/2)
            pushfirst!(outputs, "Nothing --> $(S2[j-1])")
            j = j-1
        else
            pushfirst!(outputs, "$(S1[i-1]) ---> $(S2[j-1])")
            i = i-1; j = j-1
        end
    end
    # @show outputs
    title = "\nOptimal Matching Print-out for Fixed-Penalty GED"
    println(title)
    println("-"^length(title), "\n")
    println("The cheapest way to do the tranformation...\n")
    println(S1, "---->", S2)
    println("\n...is the following series of edits...\n")
    for statement in outputs
        println(statement)
    end
end




"""
`Distance.pairwise(d::Metric, a::Vector{InteractionSequence{T}}) where {T<:Union{Int,String}})`

Distance matrix calculation between elements of Vectors. This is a custom extension
of the function in the Distances.jl package to allow vectors of general type. The
function in Distances.jl is designed for univariate/multivariate data and so takes
as input either vectors or matrices (data points as rows).

This version takes a single vector and will evaluate all pairwise distances
"""
function Distances.pairwise(
    d::Union{InteractionSeqDistance,InteractionSetDistance},
    a::Vector{InteractionSequence{T}} where {T<:Union{Int,String}}
    )

    D = zeros(length(a), length(a))
    for j in 1:length(a)
        for i in (j+1):length(a)
            tmp = d(a[i],a[j])
            D[i,j] = tmp 
            D[j,i] = tmp
        end
    end
    return D
end