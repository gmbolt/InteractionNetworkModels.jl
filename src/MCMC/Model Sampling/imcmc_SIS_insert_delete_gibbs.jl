using Distributions, StatsBase, Distributed

export draw_sample, draw_sample!, rand_multinomial_init
 

# Gibbs Move 
# ----------

function imcmc_gibbs_update!(
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    i::Int,
    model::SIS{T}, 
    mcmc::SisMcmcInsertDeleteGibbs{T}
    ) where {T<:Union{Int, String}}
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


    delete_insert!(S_prop[i], δ, d, ind_del, ind_add, vals)

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
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    model::SIS{T}, 
    mcmc::SisMcmcInsertDeleteGibbs
    ) where {T<:Union{Int, String}} 
    count = 0
    N = length(S_curr)
    for i = 1:N
        count += imcmc_gibbs_update!(S_curr, S_prop, i, model, mcmc)
    end 
    return count
end

# Trans-dimensional Move 
# ----------------------

function migrate!(
    y::Vector{Vector{Int}}, x::Vector{Vector{Int}},
    j::Int, i::Int)
    insert!(y, j, x[i])
    deleteat!(x, i)
end 

function imcmc_multi_insert_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    model::SIS{T},
    mcmc::Union{SisMcmcInsertDeleteGibbs{T},SisMcmcInsertDeleteEdit{T}},
    ind::AbstractVector{T}
    ) where {T<:Union{Int, String}}

    prop_pointers = mcmc.prop_pointers
    ν_trans_dim = mcmc.ν_trans_dim
    N = length(S_curr)
    path_dist = mcmc.path_dist

    log_ratio = 0.0 
    for i in ind 
        migrate!(S_prop, prop_pointers, i, 1)
        rand!(S_prop[i], path_dist)
        log_ratio += - logpdf(path_dist, S_prop[i])
    end 
    log_ratio += log(ν_trans_dim) - log(min(ν_trans_dim,N)) 
    return log_ratio 

end 

function imcmc_multi_delete_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T}, 
    model::SIS{T},
    mcmc::Union{SisMcmcInsertDeleteGibbs{T}, SisMcmcInsertDeleteEdit{T}},
    ind::AbstractVector{T}
    ) where {T<:Union{Int,String}}

    prop_pointers = mcmc.prop_pointers
    ν_trans_dim = mcmc.ν_trans_dim
    N = length(S_curr)
    K_outer = model.K_outer
    path_dist = mcmc.path_dist

    log_ratio = 0.0

    for i in Iterators.reverse(ind)
        migrate!(prop_pointers, S_prop, 1, i)
        log_ratio += logpdf(path_dist, S_curr[i])
    end 

    log_ratio += log(min(ν_trans_dim,N)) - log(ν_trans_dim)
    return log_ratio

end 

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T}, 
    model::SIS{T}, 
    mcmc::Union{SisMcmcInsertDeleteGibbs{T},SisMcmcInsertDeleteEdit{T}}
    )  where {T<:Union{Int, String}}

    K_outer = model.K_outer
    ν_trans_dim = mcmc.ν_trans_dim
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers

    log_ratio = 0.0

    # Enact insertion / deletion 
    N = length(S_curr)
    is_insert = rand(Bernoulli(0.5))
    if is_insert
        ε = rand(1:ν_trans_dim) # How many to insert 
        # Catch invalid proposal (ones which have zero probability)
        if (N + ε) > K_outer
            # Make no changes and imediately reject  
            return 0  
        end 
        ind_tr_dim = view(mcmc.ind_trans_dim, 1:ε) # Storage for where to insert 
        StatsBase.seqsample_a!(1:(N+ε), ind_tr_dim) # Sample where to insert 
        log_ratio += imcmc_multi_insert_prop_sample!(
            S_curr, S_prop, 
            model, mcmc, 
            ind_tr_dim
            ) # Enact move and catch log ratio term 
    else 
        ε = rand(1:min(ν_trans_dim, N)) # How many to delete
        # Catch invalid proposal (would go to empty inter seq)
        if ε == N 
            return 0 
        end  
        ind_tr_dim = view(mcmc.ind_trans_dim, 1:ε) # Storage
        StatsBase.seqsample_a!(1:N, ind_tr_dim) # Sample which to delete 
        log_ratio += imcmc_multi_delete_prop_sample!(
            S_curr, S_prop, 
            model, mcmc, 
            ind_tr_dim
            ) # Enact move and catch log ratio 
    end 


    # Now do accept-reject step 
    log_α = - model.γ * (
        model.dist(model.mode, S_prop) - model.dist(model.mode, S_curr)
    ) + log_ratio

    # Note that we copy interactions between S_prop (resp. S_curr) and prop_pointers (resp .curr_pointers) by hand.
    if log(rand()) < log_α
        if is_insert
            for i in ind_tr_dim
                migrate!(S_curr, curr_pointers, i, 1)
                copy!(S_curr[i], S_prop[i])
            end 
        else 
            for i in Iterators.reverse(ind_tr_dim)
            migrate!(curr_pointers , S_curr, 1, i)
            end 
        end 
        return 1
    else 
        # Here we must delete the interactions which were added to S_prop
        if is_insert
            for i in Iterators.reverse(ind_tr_dim)
                migrate!(prop_pointers, S_prop, 1, i)
            end 
        # Or reinsert the interactions which were deleted 
        else 
            for i in ind_tr_dim
                migrate!(S_prop, prop_pointers, i, 1)
                copy!(S_prop[i], S_curr[i])
            end 
        end 
        return 0
    end 

end 

function draw_sample!(
    sample_out::Union{InteractionSequenceSample{T}, SubArray},
    mcmc::SisMcmcInsertDeleteGibbs,
    model::SIS{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}}

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
    lb(x::Vector{Path{T}}) = max(1, length(x) - 1 )
    ub(x::Vector{Path{T}}) = min(model.K_outer, length(x) + 1)


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
    mcmc::SisMcmcInsertDeleteGibbs{T},
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}} 
    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    # @show sample_out
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 


function (mcmc::SisMcmcInsertDeleteGibbs{T})(
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
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
