export acceptance_prob

abstract type McmcMove end 

acceptance_prob(move::McmcMove) = move.counts[1]/move.counts[2]

function reset_counts!(mcmc::McmcMove)
    mcmc.counts .= 0
end 