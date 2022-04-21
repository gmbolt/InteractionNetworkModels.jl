using Distributions

export acceptance_prob

abstract type InvMcmcMove end

acceptance_prob(move::InvMcmcMove) = move.counts[1]/move.counts[2]

function reset_counts!(mcmc::InvMcmcMove)
    mcmc.counts .= 0
end 

# Any move must have the following 
# prop_sample!(S_curr,S_prop,move) (returs log ratio)
# accept_reject!(S_curr,S_prop,model,move) (returns true false)
# counts field 

# The most general enact accept/reject functions 
# These can be specialised for a move if perhaps one can work out 
# how to make the required changes more efficiently
function enact_accept!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    pointers::InteractionSequence{Int},
    move::InvMcmcMove
    )
    ε = length(S_prop) - length(S_curr)
    if ε > 0 
        # We've increased dimension
        # Bring in pointers 
        for i in 1:ε
            tmp = pop!(pointers)
            push!(S_curr,tmp)
        end 
    else 
        # We've decreased dimension
        # Send pointers back 
        for i in 1:abs(ε)
            tmp = pop!(S_curr)
            push!(pointers, tmp)
        end 
    end 
    # Copy across from S_prop 
    for (x,y) in zip(S_curr,S_prop)
        copy!(x, y)
    end 
end 

function enact_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    pointers::InteractionSequence{Int},
    move::InvMcmcMove
    )

    ε = length(S_prop) - length(S_curr)
    if ε > 0
        # We've increased dimension
        # Send back pointers 
        for i in 1:ε
            tmp = pop!(S_prop)
            push!(pointers, tmp)
        end 
    else 
        # We've decreased dimension
        # Bring in pointers 
        for i in 1:abs(ε)
            tmp = pop!(pointers)
            push!(S_prop, tmp)
        end 
    end 
    # Copy across from S_curr
    for (x,y) in zip(S_prop,S_curr)
        copy!(x, y)
    end 
end 

# Most general accept reject function 
