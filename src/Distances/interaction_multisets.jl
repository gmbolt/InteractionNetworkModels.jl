using StatsBase, Distnaces, Hungarian

export InteractionSetDistance, LengthDistance
export MatchingDist, FastMatchingDist, FpMatchingDist, FpMatchingDist2
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
            Λ = T[]
            C = [C hcat(fill([d.ground_dist(Λ, p) for p in S1], length(S1)-length(S2))...)]
            # @show C, ext_C
            return hungarian(C)[2]
        end 
    end
end

# Get MatchingDist matching

# function get_matching(d::MatchingDist, S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
#     C = Distances.pairwise(d.ground_dist, S1, S2)
#     if length(S1) == length(S2)
#         ext_C = copy(C)
#         Tplan = PythonOT.emd([], [], ext_C)
#     elseif length(S1) > length(S2)
#         # ext_C = hcat(C, hcat(fill([d.ground_dist(Path([]), p) for p in S1], length(S1)-length(S2))...)) # Extend the cost matrix
#         ext_C = hcat(C, [d.ground_dist(Path([]), S1[i]) for i=1:length(S1), j=1:(length(S1)-length(S2))])
#         Tplan = PythonOT.emd([], [], ext_C)
#     else
#         ext_C = vcat(C, [d.ground_dist(Path([]), S2[j]) for i=1:(length(S2)-length(S1)), j=1:length(S2)])
#         Tplan = PythonOT.emd([], [], ext_C)
#     end
#     for i in 1:size(ext_C)[1]
#         if i ≤ length(S1)
#             j = findfirst(Tplan[i,:].>0.0)
#             if j > length(S2)
#                 println("$(S1[i]) --> Nothing")
#             else 
#                 println("$(S1[i]) --> $(S2[j])")
#             end 
#         else 
#             j = findfirst(Tplan[i,:].>0.0)
#             if j > length(S2)
#                 println("Nothing --> Nothing")
#             else 
#                 println("Nothing --> $(S2[j])")
#             end 
#         end
#     end 
# end 


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
        for j in 1:length(S2)
            for i in 1:length(S1)
                @views C[i,j] = d.ground_dist(S1[i], S2[j])
            end 
        end 
        if length(S1) == length(S2)
            return hungarian(C)[2]
        else 
            Λ = T[]
            for i in 1:length(S1)
                C[i,length(S2)+1] = d.ground_dist(Λ, S1[i])
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

    if length(S1) < length(S2)
        d(S2,S1)
    else
        # Coming in (due to above if statement) we know S1 is longer
        C = fill(d.penalty, length(S1), length(S1))
        # @show C
        @views Distances.pairwise!(C, d.ground_dist, S1, S2)
        # @show C

        return hungarian(C)[2]
    end
end

# This is a more general approach which should be able to provide solutions for any 
# choice of penatly. 
struct FpMatchingDist2{T<:InteractionDistance} <: InteractionSetDistance
    ground_dist::T
    penalty::Real
end 

function (d::FpMatchingDist2)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

    C = fill(d.penalty, length(S1)+length(S2), length(S1)+length(S2))

    @views pairwise!(C[1:length(S1), 1:length(S2)], d.ground_dist, S1, S2)
    @views fill!(C[(end-length(S2)+1):end, (end-length(S1)+1):end], 0.0) 
    assignment, cost = hungarian(C)
    # for i = 1:(length(S1)+length(S2))
    #     j = findfirst(TransPlan[i,:] .> 0)
    #     if (j > length(S2)) & (i > length(S1)) 
    #         println("Nothing ---> Nothing, at cost $(C[i,j])")
    #     elseif (j > length(S2))
    #         println("$(S1[i]) ---> Nothing, at cost $(C[i,j])")
    #     elseif (i > length(S1) )
    #         println("Nothing ---> $(S2[j]), at cost $(C[i,j])")
    #     else
    #         println("$(S1[i]) ---> $(S2[j]), at cost $(C[i,j])")
    #     end
    # end
    return cost

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