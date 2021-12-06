using StatsBase, Distances, Printf

export InteractionSeqDistance
export EditDistance, FastEditDistance, FpEditDistance, NormFpEditDistance, AvgSizeFpEditDistance, DTW, PenalisedDTW
export print_matching
export NormPenalisedDTW

abstract type InteractionSeqDistance <: Metric end

# EditDistance
# ---
struct EditDistance{T<:InteractionDistance} <:InteractionSeqDistance
    ground_dist::T
end

function (d::EditDistance)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
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
                                        prev_row[j+1] + d₀(nothing, S1[i]),
                                        curr_row[j] + d₀(nothing, S2[j]))
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[end]
    end
end

# EditDistance with Memory
# ---------------

struct FastEditDistance{T<:InteractionDistance} <:InteractionSeqDistance
    ground_dist::T
    curr_row::Vector{Float64}
    prev_row::Vector{Float64}
    function FastEditDistance(ground_dist::S, K::Int) where {S<:InteractionDistance}
        new{S}(ground_dist, zeros(Float64, K), zeros(Float64, K))
    end 
end
function Base.show(io::IO, d::FastEditDistance)
    print(io, "EditDistance (max num. interactions $(length(d.curr_row))) with $(d.ground_dist) ground distance.")
end

function (d::FastEditDistance)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        d₀ = d.ground_dist
        prev_row = view(d.prev_row, 1:(length(S2)+1))
        curr_row = view(d.curr_row, 1:(length(S2)+1))
        # prev_row = d.prev_row
        # curr_row = d.curr_row

        prev_row[1] = 0.0
        for i in 1:length(S2)
            prev_row[i+1] = prev_row[i] + d₀(nothing, S2[i])
        end 
        curr_row .= 0.0
        

        @views for i = 1:length(S1)
            curr_row[1] = prev_row[1] + length(S1[i])
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                curr_row[j+1] = min(
                    prev_row[j] + d₀(S1[i], S2[j]), 
                    prev_row[j+1] + d₀(nothing, S1[i]), 
                    curr_row[j] + d₀(nothing, S2[j])
                    )
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[length(S2) + 1]
    end
end


function print_matching(
    d::Union{EditDistance,FastEditDistance}, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}
    
    d₀ = d.ground_dist
    # First find the substitution matrix
    C = zeros(Float64, length(S1)+1, length(S2)+1)
    C[:,1] = pushfirst!(cumsum([d₀(Λ, p) for p in S1]), 0.0);
    C[1,:] = pushfirst!(cumsum([d₀(Λ, p) for p in S2]), 0.0);

    for j in 1:length(S2)
        for i in 1:length(S1)
            C[i+1,j+1] = minimum([
                C[i,j] + d₀(S1[i], S2[j]),
                C[i,j+1] + d₀(nothing, S2[j]),
                C[i+1,j] + d₀(nothing, S1[i])
            ])
        end 
    end
    # Now retrace steps to determine an optimal matching
    i, j = size(C)
    pairs = Tuple{Int,Int}[]
    while (i ≠ 1) & (j ≠ 1)
        if C[i,j] == (C[i,j-1] + d₀(Λ, S2[j-1]))
            pushfirst!(pairs, (0,j-1))
            j = j-1
        elseif C[i,j] == (C[i-1,j] + d₀(Λ, S1[i-1]))
            pushfirst!(pairs, (i-1,0))
            i = i-1
        else
            pushfirst!(pairs,(i-1,j-1))
            i = i-1; j = j-1
        end
    end
    for k in Iterators.reverse(1:(i-1))
        pushfirst!(pairs, (k,0))
    end 
    for k in Iterators.reverse(1:(j-1))
        pushfirst!(pairs, (0,k))
    end 
    max_len = maximum(map(x->length(@sprintf("%s",x)), S1))
    # @show outputs
    title = "\nOptimal Matching"
    println(title)
    println("-"^length(title), "\n")
    for (k,l) in pairs 
        if k == 0 
            tmp_S1 = "Λ"
            tmp_S2 = S2[l] 
        elseif l == 0 
            tmp_S2 = "Λ"
            tmp_S1 = S1[k]
        else 
            tmp_S1, tmp_S2 = (S1[k], S2[l])
        end 
        pad = max_len - length(@sprintf("%s", tmp_S1))
        println("$tmp_S1" * " "^pad, " → $tmp_S2")
    end

end 


struct FpEditDistance{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real
end

function (d::FpEditDistance)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
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



function print_matching(d::FpEditDistance, S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

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
    title = "\nOptimal Matching Print-out for Fixed-Penalty EditDistance"
    println(title)
    println("-"^length(title), "\n")
    println("The cheapest way to do the tranformation...\n")
    println(S1, "---->", S2)
    println("\n...is the following series of edits...\n")
    for statement in outputs
        println(statement)
    end
end




struct AvgSizeFpEditDistance{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real
end 

function (d::AvgSizeFpEditDistance)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}

    d_ed = FpEditDistance(d.ground_dist, d.ρ)(S1, S2)

    return d_ed + (mean(length.(S1)) - mean(length.(S2)))^2

end 

# Normed EditDistance

struct NormFpEditDistance{T<:InteractionDistance} <: InteractionSeqDistance
    ground_dist::T
    ρ::Real # Penalty
end

function (d::NormFpEditDistance)(S1::InteractionSequence{T}, S2::InteractionSequence{T}) where {T<:Union{Int,String}}
    if length(S1) < length(S2)
        d(S2, S1)
    else
        tmp_d = FpEditDistance(d.ground_dist, d.ρ)(S1, S2)
        return 2*tmp_d / ( d.ρ*(length(S1) + length(S2)) + tmp_d )
    end
end

# Dynamic Time Warping (DTW)
# ==========================

# Standard 
# --------
struct DTW <:InteractionSeqDistance
    ground_dist::InteractionDistance
end

function (d::DTW)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        @assert (length(S1)>0) & (length(S2)>0) "both args must be either of type Nothing or of nonzero length."
        d_g = d.ground_dist
        prev_row = pushfirst!(fill(Inf, length(S2)), 0.0);
        curr_row = fill(Inf, length(S2) + 1);

        for i = 1:length(S1)
            # curr_row[1] = prev_row[1]
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                cost = d_g(S1[i],S2[j])
                curr_row[j+1] = cost + min(prev_row[j], prev_row[j+1], curr_row[j])
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[end]
    end
end

function (d::DTW)(
    S1::Nothing, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    d_g = d.ground_dist
    z = 0.0
    for x in S2
        z += d_g(x, nothing) 
    end 
    return z + length(S2) 

end 
(d::DTW)(S1::InteractionSequence{T}, S2::Nothing) where {T<:Union{Int,String}} = d(S2,S1)
(d::DTW)(S1::Nothing, S2::Nothing) where {T<:Union{Int,String}} = 0.0


function print_matching(
    d::DTW, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}
    
    d = d.ground_dist
    # First find the substitution matrix
    C = fill(Inf, length(S1)+1, length(S2)+1)
    C[1,1] = 0.0

    for j in 1:length(S2)
        for i in 1:length(S1)
            cost = d(S1[i], S2[j])
            C[i+1,j+1] = cost + min(C[i+1,j], C[i,j+1], C[i,j])
        end 
    end
    # Now retrace steps to determine an optimal matching
    i, j = size(C)
    pairs = Tuple{Int,Int}[]
    pushfirst!(pairs, (i-1,j-1))
    while (i ≠ 2) | (j ≠ 2)
        i_tmp, j_tmp = argmin(view(C, (i-1):i, (j-1):j)).I
        i = i - 2 + i_tmp
        j = j - 2 + j_tmp
        pushfirst!(pairs, (i-1,j-1))
    end
    # @show outputs
    title = "Optimal Coupling"
    println(title)
    println("-"^length(title), "\n")
    # for statement in outputs
    #     println(statement)
    # end
    i_tmp,j_tmp = (0,0)
    for (i,j) in pairs 
        if i == i_tmp 
            println(" "^length(@sprintf("%s",S1[i])) * " ↘ $(S2[j])")
        elseif j == j_tmp
            println("$(S1[i]) ↗ " * " "^length(@sprintf("%s",S2[j])))
        else 
            println("$(S1[i]) → $(S2[j])")
        end 
        i_tmp, j_tmp = (i,j)
    end 

end 

# Penalised 
# ---------

struct PenalisedDTW <:InteractionSeqDistance
    ground_dist::InteractionDistance
    ρ::Real
end

function (d::PenalisedDTW)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}
    if length(S1) < length(S2)  # This ensures first seq is longest
        d(S2, S1)
    else
        d_g = d.ground_dist
        prev_row = pushfirst!(fill(Inf, length(S2)), 0.0);
        curr_row = fill(Inf, length(S2) + 1);

        for i = 1:length(S1)
            # curr_row[1] = prev_row[1]
            for j = 1:(length(S2))
                # @show i, j, prev_row[j], d.ground_dist(S1[i], S2[j])
                cost = d_g(S1[i],S2[j])
                curr_row[j+1] = cost + min(
                    prev_row[j], # New pair (no warping)
                    prev_row[j+1] + d.ρ, # Warping 
                    curr_row[j] + d.ρ)  # Warping
            end
            # @show curr_row
            copy!(prev_row, curr_row)
        end
        return curr_row[end]
    end
end

function (d::PenalisedDTW)(
    S1::Nothing, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    d_g = d.ground_dist
    z = 0.0
    for x in S2
        z += d_g(x, nothing) 
    end 
    return z + length(S2) * d.ρ

end 
(d::PenalisedDTW)(S1::InteractionSequence{T}, S2::Nothing) where {T<:Union{Int,String}} = d(S2,S1)
(d::PenalisedDTW)(S1::Nothing, S2::Nothing) where {T<:Union{Int,String}} = 0.0#

struct NormPenalisedDTW <:InteractionSeqDistance
    ground_dist::InteractionDistance
    ρ::Real
    d_unnorm::PenalisedDTW
    function NormPenalisedDTW(d::InteractionDistance, ρ::Real)
        new(d, ρ, PenalisedDTW(d, ρ))
    end 
end

function (d::NormPenalisedDTW)(
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}

    d_tmp = d.d_unnorm(S1,S2)
    d_01 = d.d_unnorm(S1, nothing)
    d_02 = d.d_unnorm(S2, nothing)
    return 2 * d_tmp / (d_01 + d_02 + d_tmp)
end 

(d::NormPenalisedDTW)(S1::InteractionSequence{T}, S2::Nothing) where {T<:Union{Int,String}} = 1.0
(d::NormPenalisedDTW)(S1::Nothing, S2::InteractionSequence{T}) where {T<:Union{Int,String}} = 1.0
(d::NormPenalisedDTW)(S1::Nothing, S2::Nothing) where {T<:Union{Int,String}} = 0.0


function print_matching(
    d::Union{PenalisedDTW,NormPenalisedDTW}, 
    S1::InteractionSequence{T}, S2::InteractionSequence{T}
    ) where {T<:Union{Int,String}}
    
    d_g = d.ground_dist
    # First find the substitution matrix
    C = fill(Inf, length(S1)+1, length(S2)+1)
    C[1,1] = 0.0

    for j in 1:length(S2)
        for i in 1:length(S1)
            cost = d_g(S1[i], S2[j])
            C[i+1,j+1] = cost + min(C[i+1,j] + d.ρ, C[i,j+1] + d.ρ, C[i,j])
        end 
    end
    # Now retrace steps to determine an optimal matching
    i, j = size(C)
    pairs = Tuple{Int,Int}[]
    warp_shift_mat = [0 d.ρ; d.ρ 0] # Extra bit here different from DTW
    pushfirst!(pairs, (i-1,j-1))
    while (i ≠ 2) | (j ≠ 2)
        i_tmp, j_tmp = argmin(view(C, (i-1):i, (j-1):j) + warp_shift_mat).I
        i = i - 2 + i_tmp
        j = j - 2 + j_tmp
        pushfirst!(pairs, (i-1,j-1))
    end
    # @show outputs
    title = "Fixed Penalty DTW Print-out with $d_g Ground Distance and ρ=$(d.ρ)"
    println(title)
    println("-"^length(title), "\n")
    println("The cheapest way to do the tranformation...\n")
    println(S1, "---->", S2)
    println("\n...is the following series of edits...\n")
    # for statement in outputs
    #     println(statement)
    # end
    i_tmp,j_tmp = (0,0)
    for (i,j) in pairs 
        if i == i_tmp 
            println(" "^length(@sprintf("%s",S1[i])) * " ↘ $(S2[j])")
        elseif j == j_tmp
            println("$(S1[i]) ↗ " * " "^length(@sprintf("%s",S2[j])))
        else 
            println("$(S1[i]) → $(S2[j])")
        end 
        i_tmp, j_tmp = (i,j)
    end 

end 