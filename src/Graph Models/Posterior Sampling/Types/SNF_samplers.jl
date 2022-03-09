export SnfExRandGibbs

struct SnfExRandGibbs
    aux_mcmc::SnfMcmcSampler
    ν::Int 
    desired_samples::Int
    burn_in::Int 
    lag::Int 
end 

function SnfExRandGibbs(
    aux_mcmc; 
    ν::Int=3,
    desired_samples::Int=1000,
    burn_in::Int=0, 
    lag::Int=1
    )
    return SnfExRandGibbs(aux_mcmc, ν, desired_samples, burn_in, lag)
end 