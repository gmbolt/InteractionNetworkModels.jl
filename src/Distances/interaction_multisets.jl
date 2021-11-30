using StatsBase, Distances, Hungarian

export InteractionSetDistance, LengthDistance
export MatchingDist, FastMatchingDist, FpMatchingDist, print_matching
export CouplingDistance, PenalisedCouplingDistance
export AvgSizeFpMatchingDist, NormFpMatchingDist

abstract type InteractionSetDistance <: Metric end

# Matching Distance 
# -----------------

struct MatchingDist{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
end

function (d::MatchingDist)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int, String}}
    if length(S1) < length(S2)  # Ensure first is seq longest
        d(S2,S1)
    else
        C = Distances.pairwise(d.ground_dist, S1, S2)
        if length(S1) == length(S2)
            return hungarian(C)[2]
        else 
            null_dists = [d.ground_dist(nothing, p) for p in S1]
            size_diff = length(S1)-length(S2)
            C = [C [x for x∈null_dists, j=1:size_diff]]
            return hungarian(C)[2]
        end 
    end
end

function print_matching(
    d::MatchingDist, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    # N.B. - do not use any if statments here like in evaluation of the distance. We simply do most general formulation since this will be right in all cases, and we do not care so much for controlling the size of the optimisation problem for this function since its performance is not of a concern (purely for extra info on distance).
    C = Distances.pairwise(d.ground_dist, S1, S2)
    size_diff = length(S1)-length(S2)
    if size_diff > 0 
        null_dists = [d.ground_dist(nothing, p) for p in S1]
        C = [C [x for x∈null_dists, j=1:size_diff]]
    else 
        null_dists = [d.ground_dist(nothing, p) for p in S2]
        C = [C ;[x for j=1:(-size_diff), x∈null_dists]]
        # @show C, ext_C
    end 
    assignment, cost = hungarian(C)

    for i in 1:length(S1)
        j = assignment[i]
        if j > length(S2)
            println("$(S1[i]) ---> Null")
        else 
            println("$(S1[i]) ---> $(S2[j])")
        end 
    end 
    for j in assignment[(length(S1)+1):end]
        if j ≤ length(S2)
            println("Null ---> $(S2[j])")
        end 
    end 
end



struct FastMatchingDist <: InteractionSetDistance
    ground_dist::InteractionDistance
    C::Matrix{Float64}
    function FastMatchingDist(ground_dist::InteractionDistance, K::Int)
        new(ground_dist, zeros(K,K))
    end 
end 

function (d::FastMatchingDist)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int, String}}
    if length(S1) < length(S2)  # Ensure first is seq longest
        d(S2,S1)
    else
        C = view(d.C, 1:length(S1), 1:length(S1)) # S1 is the longest
        pairwise!(C, d.ground_dist, S1, S2)
        if length(S1) == length(S2)
            return hungarian(C)[2]
        else 
            for i in 1:length(S1)
                C[i,length(S2)+1] = d.ground_dist(nothing, S1[i])
            end 
            for j in (length(S2)+2):length(S1)
                for i in 1:length(S1)
                    C[i,j] = C[i,j-1]
                end 
            end 
            return hungarian(C)[2]
        end 
    end
end


# Fixed Penalty Matching Distance(s)
# ----------------------------------

# For when ρ > K/2, where K is maximum distance between any pair of interactions (at least between the two observations). This will always be a complete matching. 
struct FpMatchingDist{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
    penalty::Real
end

function (d::FpMatchingDist)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    length(S1) < length(S2) ? (N_min, N_max)=(length(S1),length(S2)) : (N_min,N_max)=(length(S2),length(S1))

    C = fill(d.penalty, N_max, N_max)
    @views Distances.pairwise!(C, d.ground_dist, S1, S2)
    
    # If we have max_dist > 2*ρ we must extend C to allow for interactions in BOTH observations being un-matched.
    if maximum(view(C, 1:length(S1), 1:length(S2))) > 2*d.penalty
        C = [
            C fill(d.penalty, N_max, N_min); 
            fill(d.penalty, N_min, N_max) fill(0.0, N_min, N_min)
            ]
    end  
    return hungarian(C)[2]
end


function print_matching(
    d::FpMatchingDist, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    # N.B. - do not use any if statments here like in evaluation of the distance. We simply do most general formulation since this will be right in all cases, and we do not care so much for controlling the size of the optimisation problem for this function since its performance is not of a concern (purely for extra info on distance).
    C = fill(d.penalty, length(S1)+length(S2), length(S1)+length(S2))
    @views pairwise!(C[1:length(S1), 1:length(S2)], d.ground_dist, S1, S2)
    @views fill!(C[(end-length(S2)+1):end, (end-length(S1)+1):end], 0.0) 
    assignment, cost = hungarian(C)

    for i in 1:length(S1)
        j = assignment[i]
        if j > length(S2)
            println("$(S1[i]) ---> Null")
        else 
            println("$(S1[i]) ---> $(S2[j])")
        end 
    end 
    for j in assignment[(length(S1)+1):end]
        if j ≤ length(S2)
            println("Null ---> $(S2[j])")
        end 
    end 

end


struct AvgSizeFpMatchingDist{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
    penalty::Real
end 

function (d::AvgSizeFpMatchingDist)(
    S1::Vector{Path{T}}, S2::Vector{Path{T}}
    ) where {T<:Union{Int, String}}

    d_m = FpMatchingDist(d.ground_dist, d.penalty)(S1, S2)[1]

    return d_m + (mean(length.(S1)) - mean(length.(S2)))^2
end 

struct NormFpMatchingDist{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
    penalty::Real
end

function (d::NormFpMatchingDist)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)
        d(S2,S1)
    else
        tmp_d = FpMatchingDist(d.ground_dist, d.penalty)(S1,S2)
        # @show tmp_d
        return 2*tmp_d / ( d.penalty*(length(S1) + length(S2)) + tmp_d )
    end
end

# One-sided Coupling Distance(s)
# ------------------------------

struct CouplingDistance{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
end 

function (d::CouplingDistance)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    if length(S1) < length(S2)  # Ensure first is seq longest
        d(S2,S1)
    else
        C = Distances.pairwise(d.ground_dist, S1, S2)
        if length(S1) == length(S2)
            return hungarian(C)[2]
        else 
            min_dists = minimum(C, dims=2)
            size_diff = length(S1)-length(S2)
            # C = [C [x for x∈min_dists, j=1:size_diff]]
            C = [C hcat([min_dists for i in 1:size_diff]...)]
            return hungarian(C)[2]
        end 
    end
end 

struct PenalisedCouplingDistance{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
    ρ::Real
end 

function (d::PenalisedCouplingDistance)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    d_tmp = CouplingDistance(d.ground_dist)(S1,S2)
    return d_tmp + d.ρ * abs(length(S1)-length(S2))

end 

function print_matching(
    d::Union{CouplingDistance, PenalisedCouplingDistance}, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    C = Distances.pairwise(d.ground_dist, S1, S2)
    N,M = (length(S1), length(S2))
    size_diff = N-M
    if size_diff > 0 # S1 is longer
        min_dists = minimum(C, dims=2)
        C = [C hcat([min_dists for i in 1:size_diff]...)]
    else 
        min_dists = minimum(C, dims=1)
        C = [C ;vcat([min_dists for i in 1:(-size_diff)]...)]
    end 
    assignment, cost = hungarian(C)
    
    # We line all arrows up with the longest path, so must find this length (as a string)
    max_len = maximum(map(x -> length(@sprintf("%s", x)), S1))

    # Print title 
    title = "Optimal Coupling"
    println(title)
    println("-"^length(title))

    if N ≤ M # First path is shortest (so interactions on left may pair with more than one on right)
        edge_list = [Int[] for i in 1:N]
        for i in 1:N 
            j = assignment[i]
            push!(edge_list[i], j)
        end 
        for i in (N+1):M
            j = assignment[i]
            map_from = argmin(view(C, 1:N, j))
            push!(edge_list[map_from], j)
        end 
        for i in 1:N 
            for j in 1:length(edge_list[i])
                if j == 1
                    tmp = "$(S1[i])"
                    pad = max_len - length(tmp) 
                    println(tmp * " "^pad * " → $(S2[edge_list[i][1]])")
                else 
                    println(" "^max_len * " ↘ $(S2[edge_list[i][j]])")
                end 
            end 
        end 
    else # First path is longest (so interactions on right may pair with more than one on left)
        edge_list = [Int[] for i in 1:M]
        for i in 1:N
            j = assignment[i]
            if j > M 
                map_from = argmin(view(C, i, 1:M))
                push!(edge_list[map_from], i)
            else 
                push!(edge_list[j], i)
            end 
        end 
        for i in 1:M 
            for j in 1:length(edge_list[i])
                if j == 1
                    tmp = "$(S1[edge_list[i][1]])" 
                    pad = max_len - length(tmp) 
                    println(tmp * " "^pad * " → $(S2[i])")
                else   
                    tmp = "$(S1[edge_list[i][j]])" 
                    pad = max_len - length(tmp) 
                    println(tmp * " "^pad * " ↗ ")
                end 
            end 
        end 
    end 

end

# Optimal Transport (OT) Distances 
# --------------------------------

abstract type LengthDistance <: Metric end 

struct AbsoluteDiff <: LengthDistance end 
function (d::AbsoluteDiff)(N::Int, M::Int)
    return abs(N-M)
end
struct SquaredDiff <: LengthDistance end 
function (d::SquaredDiff)(N::Int, M::Int)
    return (N-M)^2
end 

# struct EMD{T<:InteractionDistance} <: InteractionSetDistance
#     ground_dist::T
# end

# function (d::EMD)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

#     a = countmap(S1)
#     b = countmap(S2)

#     C = zeros(Float64, length(a), length(b))

#     for (j, val_b) in enumerate(keys(b))
#         for (i, val_a) in enumerate(keys(a))
#             C[i, j] = d.ground_dist(val_a, val_b)
#         end
#     end
#     p_a::Array{Float64,1} = collect(values(a))
#     p_b::Array{Float64,1} = collect(values(b))
#     p_a /= sum(p_a)
#     p_b /= sum(p_b)

#     TransPlan::Array{Float64,2} = PythonOT.emd(p_a, p_b, C)

#     # Verify not nonsense output
#     @assert(abs(sum(TransPlan) - 1.0) < 10e-5 , "Invalid transport plan, check code.")

#     return sum(TransPlan .* C)
# end


# # EMD composed with a distance of the number of interactions
# struct sEMD{T<:InteractionDistance, G<:LengthDistance} <: InteractionSetDistance
#     ground_dist::T
#     length_dist::G
#     τ::Real # Relative weighting term (a proportion weighting EMD vs length distance, high τ => high EMD weighting)
# end 

# function (d::sEMD)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int, String}}
    
#     d₁ = EMD(d.ground_dist)(S1, S2)
#     d₂ = d.length_dist(length(S1), length(S2))
#     # @show d₁, d₂

#     return d₁ + d.τ * d₂
    

# end 

# struct sEMD2{T<:InteractionDistance, G<:LengthDistance} <:InteractionSetDistance 
#     ground_dist::T
#     length_dist::G
#     τ::Real
# end 

# function (d::sEMD2)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    
#     d₁ = EMD(d.ground_dist)(S1, S2)
#     d₂ = d.length_dist(sum(length.(S1)), sum(length.(S2)))
#     # @show d₁, d₂

#     return d.τ * d₁ + (1-d.τ) * d₂
    

# end 


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