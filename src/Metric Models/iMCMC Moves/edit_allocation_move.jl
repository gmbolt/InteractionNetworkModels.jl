using Distributions

export EditAllocationMove

struct EditAllocationMove <: InvMcmcMove 
    ν::Int 
    counts::Vector{Int}  # Track acceptance 
    function EditAllocationMove(;ν::Int=3)
        new(
            ν, 
            # zeros(Int,ν),
            # [0],
            [0,0]
        )
    end 
end 

Base.show(io::IO, x::EditAllocationMove) = print(io, "EditAllocationMove(ν=$(x.ν))")

function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::EditAllocationMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
    )

    N = length(S_curr)
    δ = rand(1:move.ν)
    rem_edits = δ # Remaining edits to allocate
    len_diffs = 0
    # j = 0 # Keeps track how many interaction have been edited 
    log_prod_term = 0.0 

    for i in eachindex(S_curr)

        # If at end we just assign all remaining edits to final interaction 
        if i == N 
            δ_tmp = rem_edits
        # Otherwise we sample the number of edits via rescaled Binomial 
        else 
            p = 1/(N-i+1)
            δ_tmp = rand(Binomial(rem_edits, p)) # Number of edits to perform on ith interaction 
        end 

        if δ_tmp == 0
            continue 
        else
            # j += 1 # Increment j 
            # Make edits .... 
            @inbounds n = length(S_curr[i])
            d = rand(0:min(n-1,δ_tmp))
            m = n + δ_tmp - 2*d

            I_tmp = S_prop[i]
            rand_delete!(I_tmp, d)
            rand_insert!(I_tmp, δ_tmp-d, V)

            # push!(mode.ind_upd, i)
            # @inbounds move.ind_upd[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_prod_term += log(min(n-1, δ_tmp)+1) - log(min(m-1, δ_tmp)+1)
            len_diffs += m-n  # How much bigger the new interaction is 
        end 
        # move.num_upd[1] = j # Store number of paths updated

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 

    # # Add final part of log_ratio term
    log_ratio = log(length(V)) * len_diffs + log_prod_term

    return log_ratio

end 

# function enact_accept!(
#     S_curr::InteractionSequence{Int},
#     S_prop::InteractionSequence{Int},
#     pointers::InteractionSequence{Int},
#     move::EditAllocationMove
#     )
#     ind = view(move.ind_upd, 1:move.num_upd[1])
#     for i in ind
#         @inbounds copy!(S_curr[i], S_prop[i])
#     end
# end 

# function enact_reject!(
#     S_curr::InteractionSequence{Int},
#     S_prop::InteractionSequence{Int},
#     pointers::InteractionSequence{Int},
#     move::EditAllocationMove
#     )
#     ind = view(move.ind_upd, 1:move.num_upd[1])
#     for i in ind
#         @inbounds copy!(S_prop[i], S_curr[i])
#     end
# end 