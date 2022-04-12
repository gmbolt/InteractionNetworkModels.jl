export InsertDeleteMove

struct InsertDeleteMove <: InvMcmcMove 
    ν::Int 
    len_dist::DiscreteUnivariateDistribution
    counts::Vector{Int}  
    function InsertDeleteMove(
        ;ν::Int=1, 
        len_dist::DiscreteUnivariateDistribution=Geometric(0.9)
        )
        new(
            ν, len_dist, 
            # zeros(Int, ν), zeros(Int, ν), [0,0],
            [0,0]
        )
    end 
end 

Base.show(io::IO, x::InsertDeleteMove) = print(io, "InsertDeleteMove(ν=$(x.ν), len_dist=$(x.len_dist))")


function multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::InsertDeleteMove,
    pointers::InteractionSequence{Int},
    ε::Int,
    V::UnitRange
    )
    len_dist = move.len_dist
    log_ratio = 0.0
    # ind = move.ind_add
    
    n = length(S_prop)+ε
    k = ε
    i = 0 
    # j = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i += 1
        # j += 1
        # Insert path at index i
        # i is now index to insert
        # ind[j] = i
        tmp = popfirst!(pointers) # Get storage 
        m = rand(len_dist)  # Sample length 
        resize!(tmp, m) # Resize 
        sample!(V, tmp) # Sample new entries uniformly
        @inbounds insert!(S_prop, i, tmp) # Insert path into S_prop
        log_ratio += - logpdf(len_dist, m) + m*log(length(V))
        n -= 1
        k -= 1
    end
    return log_ratio 

end 

function multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int}, 
    S_prop::InteractionSequence{Int},
    move::InsertDeleteMove,
    pointers::InteractionSequence{Int},
    ε::Int,
    V::UnitRange
    )

    len_dist = move.len_dist
    log_ratio = 0.0
    # ind = move.ind_del # Store which entries were deleted 

    n = length(S_curr)
    k = ε   
    i = 0 
    # j = 0
    live_index = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        # j+=1
        # Delete path 
        # i is now index to delete 
        # @inbounds ind[j] = i
        @inbounds tmp = popat!(S_prop, i - live_index)
        pushfirst!(pointers, tmp)
        m = length(tmp)
        log_ratio += logpdf(len_dist, m) - m * log(length(V))
        live_index += 1
        n -= 1
        k -= 1
    end
    if length(S_curr)-ε<1
        log_ratio += -Inf
    end 
    return log_ratio

end 

function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::InsertDeleteMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
    )

    N = length(S_curr)
    ν = move.ν
    ε = rand(1:ν)
    d = rand(0:min(ε, N))
    a = ε - d 
    # move.num_add_del[1] = a
    # move.num_add_del[2] = d

    log_ratio = 0.0
    # @show d, a, S_curr, S_prop
    if d > 0 
        log_ratio += multi_delete_prop_sample!(
            S_curr, S_prop,
            move, pointers,
            d, V
        )
    end 
    if a > 0 
        log_ratio += multi_insert_prop_sample!(
            S_curr, S_prop, 
            move, pointers, 
            a, V 
        )
    end 
    
    M = N - d + a
    log_ratio += log(min(ε,N)+1) - log(min(ε, M)+1)

    return log_ratio
end 

# function enact_accept!(
#     S_curr::InteractionSequence{Int},
#     S_prop::InteractionSequence{Int},
#     pointers::InteractionSequence{Int},
#     move::InsertDeleteMove
#     )
#     ind = view(move.ind_del, 1:move.num_add_del[2])
#     for i in Iterators.reverse(ind) # If we didnt do reverse would have to update indices 
#         @inbounds tmp = popat!(S_curr, i)
#         pushfirst!(pointers, tmp)
#     end
#     ind = view(move.ind_add, 1:move.num_add_del[1])
#     for i in ind 
#         tmp = popfirst!(pointers)
#         copy!(tmp, S_prop[i])
#         insert!(S_curr, i, tmp)
#     end 
# end 

# function enact_reject!(
#     S_curr::InteractionSequence{Int},
#     S_prop::InteractionSequence{Int},
#     pointers::InteractionSequence{Int},
#     move::InsertDeleteMove
#     )
#     ind = view(move.ind_add, 1:move.num_add_del[1])
#     for i in Iterators.reverse(ind)
#         @inbounds tmp = popat!(S_prop, i)
#         pushfirst!(pointers, tmp)
#     end 
#     ind = view(move.ind_del, 1:move.num_add_del[2])
#     for i in ind 
#         @inbounds tmp = popfirst!(pointers)
#         copy!(tmp, S_curr[i])
#         insert!(S_prop, i, tmp)
#     end
# end 