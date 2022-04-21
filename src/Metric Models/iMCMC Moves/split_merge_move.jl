export SplitMergeMove 

struct SplitMergeMove <: InvMcmcMove 
    ν::Int 
    ind_split::Vector{Int}
    ind_merge::Vector{Int}
    counts::Vector{Int}
    function SplitMergeMove(
        ;ν::Int=1
        )
        ind_split = zeros(Int, ν)
        ind_merge = zeros(Int, 2ν)
        new(ν, ind_split, ind_merge, [0,0])
    end 
end 

Base.show(io::IO, x::SplitMergeMove) = print(io, "SplitMergeMove(ν=$(x.ν))")

function multi_split_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::SplitMergeMove,
    pointers::InteractionSequence{Int}
    ) 

    ν = move.ν
    N = length(S_curr)
    ε = rand(1:min(N, ν))
    ind_split = view(move.ind_split, 1:ε)
    ind_loc = view(move.ind_merge, 1:2ε)

    store = Vector{Int}[]

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_split)
    StatsBase.seqsample_a!(1:(N+ε), ind_loc)

    for i in Iterators.reverse(ind_split)
        @assert length(S_prop[i])>1 "Cannot split path of length one."
        tmp1 = popat!(S_prop, i)
        n = length(tmp1)
        k = rand(1:(n-1)) # Split location 
        log_ratio += log(n-1)
        tmp2 = popfirst!(pointers)
        copy!(tmp2, tmp1[(k+1):end])
        resize!(tmp1, k)
        pushfirst!(store, tmp2)
        pushfirst!(store, tmp1)
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(N, ν)) - log(min(floor(length(S_prop/2)), ν))

    return log_ratio 

end 


function multi_merge_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::SplitMergeMove,
    pointers::InteractionSequence{Int}
    ) 

    ν = move.ν
    N = length(S_curr)
    ε = rand(1:min(floor(Int, N/2), ν))

    ind_merge = view(move.ind_merge, 1:2ε)
    ind_loc = view(move.ind_split, 1:ε)

    store = Vector{Int}[]

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_merge)
    StatsBase.seqsample_a!(1:(N-ε), ind_loc)

    i = length(ind_merge)
    while i > 0
        # Merge i-1 and i 
        tmp2 = popat!(S_prop, ind_merge[i])
        tmp1 = popat!(S_prop, ind_merge[i-1])
        append!(tmp1, tmp2)
        pushfirst!(pointers, tmp2)
        pushfirst!(store, tmp1)
        log_ratio -= log(length(tmp1)-1)
        i -= 2 
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(floor(Int, N/2), ν)) - log(min(length(S_prop), ν))

    return log_ratio 
end 

# function log_binomial_ratio_split_merge(N, τ, ε)
#     # \binom{N-τ}{ε}/\binom{N}{ε}
#     out = 0.0
#     if τ < ε 
#         for j in 0:(τ-1)
#             out += log(N-ε-j) - log(N-j)
#         end 
#     else 
#         for j in 0:(ε-1)
#             out += log(N-τ-j) - log(N-j)
#         end 
#     end
#     return out 
# end 


# Special case (when we have paths of length one present)
function special_multi_split_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::SplitMergeMove,
    pointers::InteractionSequence{Int}
    ) 

    ν = move.ν
    N = length(S_curr)
    ind_ok_len = findall(x->length(x)>1, S_curr) # Good paths (length > 1)
    τ = N-length(ind_ok_len)                     # Number of bad paths 
    ε = rand(1:min(N-τ, ν))     
    ind_split = view(move.ind_split, 1:ε)
    ind_loc = view(move.ind_merge, 1:2ε)

    store = Vector{Int}[]

    log_ratio = 0.0

    StatsBase.seqsample_a!(ind_ok_len, ind_split)  # Sample subsequence of good paths
    StatsBase.seqsample_a!(1:(N+ε), ind_loc)

    for i in Iterators.reverse(ind_split)
        @assert length(S_prop[i])>1 "Cannot split path of length one."
        tmp1 = popat!(S_prop, i)
        n = length(tmp1)
        k = rand(1:(n-1)) # Split location 
        log_ratio += log(n-1)
        tmp2 = popfirst!(pointers)
        copy!(tmp2, tmp1[(k+1):end])
        resize!(tmp1, k)
        pushfirst!(store, tmp2)
        pushfirst!(store, tmp1)
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    log_ratio += log(min(N-τ, ν)) - log(min(floor(length(S_prop/2)), ν))

    log_ratio += log_binomial_ratio_split_merge(N, τ, ε)

    return log_ratio 

end 



function special_multi_merge_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::SplitMergeMove,
    pointers::InteractionSequence{Int}
    ) 

    ν = move.ν
    N = length(S_curr)
    ε = rand(1:min(floor(Int, N/2), ν))

    ind_merge = view(move.ind_merge, 1:2ε)
    ind_loc = view(move.ind_split, 1:ε)

    store = Vector{Int}[]

    log_ratio = 0.0

    StatsBase.seqsample_a!(1:N, ind_merge)
    StatsBase.seqsample_a!(1:(N-ε), ind_loc)

    i = length(ind_merge)
    while i > 0
        # Merge i-1 and i 
        tmp2 = popat!(S_prop, ind_merge[i])
        tmp1 = popat!(S_prop, ind_merge[i-1])
        append!(tmp1, tmp2)
        pushfirst!(pointers, tmp2)
        pushfirst!(store, tmp1)
        log_ratio -= log(length(tmp1)-1)
        i -= 2 
    end 

    for i in ind_loc
        tmp = popfirst!(store)
        insert!(S_prop, i, tmp)
    end 

    M = length(S_prop)
    τ = sum(length(x)==1 for x in S_prop) # Find number of length one paths
    log_ratio -= log_binomial_ratio_split_merge(M, τ, ε)

    log_ratio += log(min(floor(Int, N/2), ν)) - log(min(M-τ, ν))

    return log_ratio 
end 


function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::SplitMergeMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
    )

    log_ratio = 0.0 
    τ = sum(length(x)==1 for x in S_curr)
    N = length(S_curr)

    # If there is only one path of lenth one, we cannot do anything. 
    # So just reject instantly 
    if (N==τ==1)
        return 0, suff_stat_curr
    # If one path of length > 1 we split with probability one 
    elseif N==1
        do_split = true 
        log_ratio += multi_split_prop_sample!(
            S_curr, S_prop, 
            move,
            pointers    
        )
        # If S_prop has all length one paths, we would come back with probability one 
        # hence the following...
        log_ratio += all(length(x)==1 for x in S_prop) ? 0.0 : log(0.5) 
    # If no length one paths, use std proposal 
    elseif τ==0 
        do_split = rand(Bool)
        if do_split
        log_ratio += multi_split_prop_sample!(
            S_curr, S_prop, 
            move,
            pointers 
        )
        else 
            log_ratio += multi_merge_prop_sample!(
                S_curr, S_prop, 
                move,
                pointers 
            )
        end 
    # If all paths are length one, do merge with probability one
    elseif τ==N
        do_split = false
        log_ratio += special_multi_merge_prop_sample!(
            S_curr, S_prop, 
            move,
            pointers 
        )
        log_ratio += log(0.5) # Accounts for uneven choice between this and split move
    # Else to use special proposal function 
    else 
        do_split = rand(Bool)
        if do_split
            log_ratio += special_multi_split_prop_sample!(
                S_curr, S_prop, 
                move,
                pointers 
            )
        else 
            log_ratio += special_multi_merge_prop_sample!(
                S_curr, S_prop, 
                move,
                pointers 
            )
        end 
    end 
    return log_ratio
end 

