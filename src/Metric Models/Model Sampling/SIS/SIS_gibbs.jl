using Distributions, StatsBase, Distributed

export draw_sample, draw_sample!, rand_multinomial_init 

# Gibbs Move 
# ----------

function imcmc_gibbs_update!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    i::Int,
    model::SIS, 
    mcmc::SisMcmcInsertDeleteGibbs
    ) 
    # S_prop = copy(S_curr)

    # @inbounds m = rand_reflect(length(curr[i]), ν, 1, model.K_inner)

    @inbounds n = length(S_curr[i])
    δ = rand(1:mcmc.ν_gibbs)
    # a,b = (lb(n, δ, model), ub(n, δ))
    # @show n, δ, a, b
    d = rand(0:min(n,δ))
    m = n + δ - 2*d

    # Catch invalid proposal (m > K_inner). Here we imediately reject, making no changes.
    if (m > model.K_inner) | (m < 1)
        return 0 
    end 
    
    # @show m 
    # Set-up views 
    ind_del = view(mcmc.ind_del, 1:d)
    ind_add = view(mcmc.ind_add, 1:(δ-d))
    vals = view(mcmc.vals, 1:(δ-d))

    # Sample indexing info and new entries (all in-place)
    StatsBase.seqsample_a!(1:n, ind_del)
    StatsBase.seqsample_a!(1:m, ind_add)
    sample!(model.V, vals)


    delete_insert!(S_prop[i], ind_del, ind_add, vals)

    log_ratio = log(min(n, δ)+1) - log(min(m, δ)+1) + (m - n)*log(length(model.V))


    # @show curr_dist, prop_dist
    @inbounds log_α = (
        -model.γ * (
            model.dist(model.mode, S_prop)-model.dist(model.mode, S_curr)
            ) + log_ratio
        )
    if log(rand()) < log_α
        # println("accepted")
        copy!(S_curr[i], S_prop[i])
        # @inbounds S_curr[i] = copy(S_prop[i])
        # println("$(i) value proposal $(tmp_prop) was accepted")
        return 1
    else
        copy!(S_prop[i], S_curr[i])
        # @inbounds S_prop[i] = copy(S_curr[i])
        # println("$(i) value proposal $(tmp_prop) was rejected")
        return 0
    end 
    # @show S_curr
end 


function imcmc_gibbs_scan!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    model::SIS, 
    mcmc::SisMcmcInsertDeleteGibbs
    )
    count = 0
    N = length(S_curr)
    for i = 1:N
        count += imcmc_gibbs_update!(S_curr, S_prop, i, model, mcmc)
    end 
    return count
end

# Sampler Functions 
# -----------------

function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SisMcmcInsertDeleteGibbs,
    model::SIS;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(model, mcmc.init)
    ) 

    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    
    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for i in 1:length(init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], init[i])
        copy!(S_prop[i], init[i])
    end 
    # @show S_curr, S_prop, init
    # S_curr = copy(init)
    # S_prop = copy(init)

    ind = 0
    sample_count = 1
    i = 0
    gibbs_scan_count = 0
    gibbs_tot_count = 0
    gibbs_acc_count = 0 
    tr_dim_count = 0
    tr_dim_acc_count = 0

    # Bounds for uniform sampling
    lb(x::Vector{Path{Int}}) = max(1, length(x) - 1 )
    ub(x::Vector{Path{Int}}) = min(model.K_outer, length(x) + 1)


    while sample_count ≤ length(sample_out)
        i += 1
        # Store value
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end 
        # Gibbs scan
        if rand() < mcmc.β
            # println("Gibbs")
            gibbs_scan_count += 1
            gibbs_tot_count += length(S_curr)
            gibbs_acc_count += imcmc_gibbs_scan!(S_curr, S_prop, model, mcmc) # This enacts the scan, changing curr, and outputs number of accepted moves.
        # Else do insert or delete
        else 
            # println("Transdim")
            tr_dim_count += 1
            tr_dim_acc_count += imcmc_trans_dim_accept_reject!(
                S_curr, S_prop, 
                model, mcmc
            )
        end 
    end 
    # Send storage back
    # @show S_curr, S_prop
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return (
        gibbs_tot_count, gibbs_scan_count, gibbs_acc_count,
        tr_dim_count, tr_dim_acc_count
    )
    
end 

function draw_sample(
    mcmc::SisMcmcInsertDeleteGibbs,
    model::SIS;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{Int}}=get_init(model, mcmc.init)
    )
    sample_out = Vector{Vector{Path{Int}}}(undef, desired_samples)
    # @show sample_out
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 


function (mcmc::SisMcmcInsertDeleteGibbs)(
    model::SIS;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{Int}}=get_init(model, mcmc.init)
    ) 

    sample_out = Vector{Vector{Path{Int}}}(undef, desired_samples)
    # @show sample_out
    (
        gibbs_tot_count, 
        gibbs_scan_count, 
        gibbs_acc_count,
        tr_dim_count,
        tr_dim_acc_count
        ) = draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)

    p_measures = Dict(
            "Proportion Gibbs moves" => gibbs_scan_count/(tr_dim_count + gibbs_scan_count),
            "Trans-dimensional move acceptance probability" => tr_dim_acc_count/tr_dim_count,
            "Gibbs move acceptance probability" => gibbs_acc_count/gibbs_tot_count
        )
    output = SisMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 
