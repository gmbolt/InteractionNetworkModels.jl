using Distributions, StatsBase, Distributed

export imcmc_insert_prop_sample, imcmc_delete_prop_sample, draw_sample, draw_sample!
export imcmc_gibbs_update!, imcmc_gibbs_scan!, pdraw_sample, plog_aux_term_mode_update


function imcmc_insert_prop_sample(
    S_curr, path_dist
    )

    i = rand(1:(length(S_curr)+1))
    S_prop = copy(S_curr)
    I′ = rand(path_dist)
    insert!(S_prop, i, I′)
    log_ratio = - logpdf(path_dist, I′)

    return S_prop, log_ratio
end 

function imcmc_delete_prop_sample(
    S_curr, path_dist
    )

    i = rand(1:length(S_curr))
    S_prop = copy(S_curr)
    deleteat!(S_prop, i)
    log_ratio = logpdf(path_dist, S_curr[i])

    return S_prop, log_ratio
end 

function imcmc_insert_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    curr_pointers::InteractionSequence{T},
    prop_pointers::InteractionSequence{T},
    i::Int,
    path_dist::PathDistribution{T}) where {T<:Union{Int,String}}
        
    # ind = rand(1:(length(S_curr)+1))
    # S_prop = copy(S_curr)
    migrate!(S_prop, prop_pointers, i, 1) # Move new memory in 
    rand!(S_prop[i], path_dist)
    # insert!(S_prop, i, I′)
    log_ratio = - logpdf(path_dist, S_prop[i])

    return log_ratio
end 

function imcmc_delete_prop_sample!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T},
    curr_pointers::InteractionSequence{T},
    prop_pointers::InteractionSequence{T},
    i::Int,
    path_dist::PathDistribution{T}) where {T<:Union{Int,String}}

    # ind = rand(1:length(S_curr))
    # S_prop = copy(S_curr)

    migrate!(prop_pointers, S_prop, 1, i)
    # deleteat!(S_prop, i)
    log_ratio = logpdf(path_dist, S_curr[i])

    return log_ratio
end 

# Helper functions for finding upper and lower bounds for uniform sampling
lb(n::Int, δ::Int, model::Union{SIS, SIM}) = max(0, ceil(Int, (n + δ - model.K_inner)/2))
ub(n::Int, δ::Int) = min(n-1, δ)

function imcmc_gibbs_update!(
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T},
    i::Int,
    model::SIS{T}, 
    mcmc::SisInvolutiveMcmcInsertDelete{T}
    ) where {T<:Union{Int, String}}
    # S_prop = copy(S_curr)

    # @inbounds m = rand_reflect(length(curr[i]), ν, 1, model.K_inner)

    @inbounds n = length(S_curr[i])
    δ = rand(1:mcmc.ν)
    a,b = (lb(n, δ, model), ub(n, δ))
    # @show n, δ, a, b
    d = rand(a:b)
    m = n + δ - 2*d
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

    log_ratio = log(b - a + 1) - log(ub(m, δ) - lb(m, δ, model) +1) + (δ - 2*d) * log(length(model.V))


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
    mcmc::SisInvolutiveMcmcInsertDelete
    ) where {T<:Union{Int, String}} 
    count = 0
    N = length(S_curr)
    for i = 1:N
        count += imcmc_gibbs_update!(S_curr, S_prop, i, model, mcmc)
    end 
    return count
end


function migrate!(
    y::Vector{Vector{Int}}, x::Vector{Vector{Int}},
    j::Int, i::Int)
    insert!(y, j, x[i])
    deleteat!(x, i)
end 

function draw_sample!(
    sample_out::Union{InteractionSequenceSample{T}, SubArray},
    mcmc::SisInvolutiveMcmcInsertDelete,
    model::SIS{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=[rand(model.V, x) for x in length.(model.mode)]
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
    count = 0
    sample_count = 1
    i = 0
    acc_count = 0
    gibbs_scan_count = 0
    gibbs_tot_count = 0
    gibbs_acc_count = 0 
    # is_insert = false

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
            count += 1
            # If only one interaction we only insert
            if length(S_curr) == 1 
                is_insert = true
                ind = rand(1:(length(S_curr)+1))
                log_ratio = imcmc_insert_prop_sample!(
                    S_curr, S_prop, 
                    curr_pointers, prop_pointers,
                    ind, 
                    mcmc.path_dist
                )
                log_ratio += 0.5 # Adjust log ratio
                
            # If max mnumber of interactions we only delete
            elseif length(S_curr) == model.K_outer
                is_insert = false
                ind = rand(1:length(S_curr))
                log_ratio = imcmc_delete_prop_sample!(
                    S_curr, S_prop, 
                    curr_pointers, prop_pointers,
                    ind, 
                    mcmc.path_dist
                )
                log_ratio += 0.5 # Adjust log ratio

            elseif rand() < 0.5  # Coin flip for insert
                is_insert = true
                ind = rand(1:(length(S_curr)+1))
                log_ratio = imcmc_insert_prop_sample!(
                    S_curr, S_prop, 
                    curr_pointers, prop_pointers,
                    ind, 
                    mcmc.path_dist
                )
            else # Else delete
                is_insert = false
                ind = rand(1:length(S_curr))
                log_ratio = imcmc_delete_prop_sample!(
                    S_curr, S_prop, 
                    curr_pointers, prop_pointers, 
                    ind, 
                    mcmc.path_dist
                )
            end 
            # println(S_curr)
            log_α = - model.γ * (
                model.dist(model.mode, S_prop) - model.dist(model.mode, S_curr)
            ) + log_ratio

            if rand() < exp(log_α)
                if is_insert
                    migrate!(S_curr, curr_pointers, ind, 1)
                    copy!(S_curr[ind], S_prop[ind])
                    # insert!(S_curr, ind, copy(S_prop[ind]))
                else 
                    migrate!(curr_pointers, S_curr, 1, ind)
                    # deleteat!(S_curr, ind)
                end 
                acc_count += 1
            else
                if is_insert
                    migrate!(prop_pointers, S_prop, 1, ind)
                    # deleteat!(S_prop, ind)
                else 
                    migrate!(S_prop, prop_pointers, ind, 1)
                    # insert!(S_prop, ind, copy(S_curr[ind]))
                end 
            end 
            # sample_out[i] = copy(S_curr)
        end 
    end 
    # Send storage back
    # @show S_curr, S_prop
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return count, acc_count, gibbs_tot_count, gibbs_scan_count, gibbs_acc_count
    
end 

function draw_sample(
    mcmc::SisInvolutiveMcmcInsertDelete{T},
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=[rand(model.V, x) for x in length.(model.mode)]
    ) where {T<:Union{Int,String}} 
    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    # @show sample_out
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 


function (mcmc::SisInvolutiveMcmcInsertDelete{T})(
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=[rand(model.V, x) for x in length.(model.mode)]
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    # @show sample_out
    (
        count, 
        acc_count, 
        gibbs_tot_count, 
        gibbs_scan_count, 
        gibbs_acc_count
        ) = draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)

    p_measures = Dict(
            "Proportion Gibbs moves" => gibbs_scan_count/(count + gibbs_scan_count),
            "Trans-dimensional move acceptance probability" => acc_count/count,
            "Gibbs move acceptance probability" => gibbs_acc_count/gibbs_tot_count
        )
    output = SisMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 


function pdraw_sample(
    mcmc::SisInvolutiveMcmcInsertDelete{T},
    model::SIS{T}, 
    split::Vector{Int}; 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=model.mode
    ) where {T<:Union{Int,String}}
    S_init = draw_sample(mcmc, model, desired_samples=1, burn_in=burn_in, init=init)[1] 
    # @show S_init
    out = Distributed.pmap(split) do index
        return draw_sample(
                mcmc, model, 
                desired_samples=index,
                lag=lag, burn_in=0, init=S_init
                )
    end 
    vcat(out...)
end 

function pdraw_sample(
    mcmc::SisInvolutiveMcmcInsertDelete{T},
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=model.mode
) where {T<:Union{Int,String}}
    psplit = get_split(desired_samples, nworkers())
    return pdraw_sample(mcmc, model, psplit, burn_in=burn_in, lag=lag, init=init)

end 

# Parallel log auxiliary term for model update7
# - γ * (∑ d(x, S_curr) - ∑ d(x, S_prop)) (where sum is over auxiliary data)
function plog_aux_term_mode_update(
    mcmc::SisInvolutiveMcmcInsertDelete, 
    aux_model::SIS{T},
    split::Vector{T}, 
    S_curr::InteractionSequence{T},
    S_prop::InteractionSequence{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=aux_model.mode
    ) where {T<:Union{Int,String}}
    S_init = draw_sample(mcmc, aux_model, desired_samples=1, burn_in=burn_in, init=init)[1]
    out = Distributed.pmap(split) do index
        sample = draw_sample(
                mcmc, aux_model, 
                desired_samples=index,
                lag=lag, burn_in=0, init=S_init
                )
        log_lik = mapreduce(
            x -> -aux_model.γ * (aux_model.dist(x,S_curr) - aux_model.dist(x,S_prop)),
            (+),
            sample
        ) 
        return log_lik
    end 
    mapreduce(x->x[1], (+), out)
end 


