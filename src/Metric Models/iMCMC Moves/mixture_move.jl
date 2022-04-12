export InvMcmcMixtureMove

struct InvMcmcMixtureMove{N,M<:NTuple{N,InvMcmcMove}} <: InvMcmcMove
    moves::M
    p::NTuple{N,Float64}
    p_cusum::NTuple{N,Float64}
end 

InvMcmcMixtureMove(moves, p) = InvMcmcMixtureMove(moves, p, cumsum(p))

acceptance_prob(x::InvMcmcMixtureMove) = [acceptance_prob(move) for move in x.moves]
function reset_counts!(x::InvMcmcMixtureMove) 
    for move in x.moves 
        reset_counts!(move)
    end
end 

function Base.show(
    io::IO, x::InvMcmcMixtureMove
    ) 

    title = "InvMcmcMixtureMove ($(length(x.moves)) components)"
    println(io, title)
    for (p,move) in zip(x.p,x.moves)
        println(io, "   * $p - ", "$move")
    end 
end 

# We specialise the accept_reject function 
function accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    pointers::InteractionSequence{Int},
    mix_move::InvMcmcMixtureMove,
    model::T
    ) where {T<:Union{SIS,SIM}}

    # Sample move 
    p = mix_move.p_cusum  # Mixture proportions 
    u = rand()      # Random unif(0,1)
    i = findfirst(x->x>u, p)

    # Select ith move
    move = mix_move.moves[i]

    # Do accept-reject for the move
    accept_reject!(S_curr, S_prop, pointers, move, model)

end 