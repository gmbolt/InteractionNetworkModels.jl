using Distributions, StatsBase

export imcmc_multinomial_edit_accept_reject!, unif_multinomial_sample_tester
export imcmc_trans_dim_accept_reject!, draw_sample!, draw_sample

function imcmc_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence{T}, 
    S_prop::InteractionSequence{T}, 
    model::SIS{T}, 
    mcmc::SisMcmcInsertDeleteEdit{T}
    ) where {T<:Union{Int, String}}

    N = length(S_curr)  
    K_inner = model.K_inner
    δ = rand(1:mcmc.ν_edit)  # Number of edits to enact 
    rem_edits = δ # Remaining edits to allocate
    len_diffs = 0
    j = 0 # Keeps track how many interaction have been edited 
    log_prod_term = 0.0 

    # println("Making $δ edits in total...")

    for i = 1:N

        # If at end we just assign all remaining edits to final interaction 
        if i == N 
            δ_tmp = rem_edits
        # Otherwise we sample the number of edits via rescaled Binomial 
        else 
            p = 1/(N-i+1)
            δ_tmp = rand(Binomial(rem_edits, p)) # Number of edits to perform on ith interaction 
        end 

        # println("   Index $i getting $δ_tmp edits...")
        # If we sampled zero edits we skip to next iteration 
        if δ_tmp == 0
            continue 
        else
            j += 1 # Increment j 
            # Make edits .... 
            @inbounds n = length(S_curr[i])
            # a,b = (lb(n, δ_tmp, model), ub(n, δ_tmp))
            d = rand(0:min(n,δ_tmp))
            m = n + δ_tmp - 2*d

            # Catch invalid proposals
            if (m < 1) | (m > K_inner)
                # Here we just reject the proposal
                for i in 1:N
                    copy!(S_prop[i], S_curr[i])
                end 
                return 0 
            end 

            # tot_dels += d
            # println("       Deleting $d and adding $(δ_tmp-d)")
            ind_del = view(mcmc.ind_del, 1:d)
            ind_add = view(mcmc.ind_add, 1:(δ_tmp-d))
            vals = view(mcmc.vals, 1:(δ_tmp-d))

            # println("           ind_del: $ind_del ; ind_add: $ind_add")

            # Sample indexing info and new entries (all in-place)
            StatsBase.seqsample_a!(1:n, ind_del)
            StatsBase.seqsample_a!(1:m, ind_add)
            sample!(model.V, vals)

            delete_insert!(S_prop[i], δ_tmp, d, ind_del, ind_add, vals)

            mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_prod_term += log(min(n, δ_tmp)+1) - log(min(m, δ_tmp)+1)
            len_diffs += m-n  # How much bigger the new interaction is 
        end 

        # Update rem_edits
        rem_edits -= δ_tmp

        # If no more left terminate 
        if rem_edits == 0
            break 
        end 

    end 

    # # Add final part of log_ratio term
    log_ratio = log(length(model.V)) * len_diffs + log_prod_term
    # log_ratio = log_dim_diff + log_prod_term
    log_lik_ratio = -model.γ * (
        model.dist(model.mode, S_prop)-model.dist(model.mode, S_curr)
        )

    # @show log_dim_diff, log_prod_term, log_lik_ratio
        
    # Log acceptance probability
    log_α = log_lik_ratio + log_ratio

    # mean_len_prop = mean(length.(S_prop))
    # mean_len_curr = mean(length.(S_curr))

    # @show log_dim_diff, log_prod_term, log_lik_ratio, log_α, mean_len_curr, mean_len_prop

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_curr[i], S_prop[i])
        end
        return 1 
    else 
        for i in view(mcmc.ind_update, 1:j)
            copy!(S_prop[i], S_curr[i])
        end 
        return 0 
    end 
end 

# To check the above code is doing as required. Output should be samples from a uniform
# with n trials and k bins. 
function unif_multinomial_sample_tester(k, n)
    n_rem = n
    x = Int[]
    for i in 1:(k-1) 
        p = 1/(k-i+1)
        z = rand(Binomial(n_rem, p)) 
        push!(x, z)
        n_rem -= z
    end 
    push!(x, n_rem)
    return x 
end 




function draw_sample!(
    sample_out::Union{InteractionSequenceSample{T}, SubArray},
    mcmc::SisMcmcInsertDeleteEdit{T},
    model::SIS{T};
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{T}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}}



    # Define aliases for pointers to the storage of current vals and proposals
    curr_pointers = mcmc.curr_pointers
    prop_pointers = mcmc.prop_pointers
    β = mcmc.β

    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for i in 1:length(init)
        migrate!(S_curr, curr_pointers, i, 1)
        migrate!(S_prop, prop_pointers, i, 1)
        copy!(S_curr[i], init[i])
        copy!(S_prop[i], init[i])
    end 

    sample_count = 1 # Keeps which sample to be stored we are working to get 
    i = 0 # Keeps track all samples (included lags and burn_ins) 
    upd_count = 0
    upd_acc_count = 0
    tr_dim_count = 0 
    tr_dim_acc_count = 0

    while sample_count ≤ length(sample_out)
        i += 1 

        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end 
        # W.P. do update move (accept-reject done internally by function call)
        if rand() < β
            upd_acc_count += imcmc_multinomial_edit_accept_reject!(
                S_curr, S_prop, 
                model, mcmc
                )
            upd_count += 1
        # Else do trans-dim move. We will do accept-reject move here 
        else 
            tr_dim_acc_count += imcmc_trans_dim_accept_reject!(
                S_curr, S_prop, 
                model, mcmc
            )
            tr_dim_count += 1
        end 
    end 
    for i in 1:length(S_curr)
        migrate!(curr_pointers, S_curr, 1, 1)
        migrate!(prop_pointers, S_prop, 1, 1)
    end 
    return (
                upd_count, upd_acc_count,
                tr_dim_count, tr_dim_acc_count
            )
end 

function draw_sample(
    mcmc::SisMcmcInsertDeleteEdit{T}, 
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}} 

    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 

function (mcmc::SisMcmcInsertDeleteEdit{T})(
    model::SIS{T};
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Vector{Path{T}}=get_init(model, mcmc.init)
    ) where {T<:Union{Int,String}}

    sample_out = Vector{Vector{Path{T}}}(undef, desired_samples)
    # @show sample_out
    (
        update_count, update_acc_count, 
        trans_dim_count, trans_dim_acc_count
        ) = draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)

    p_measures = Dict(
            "Proportion Update Moves" => update_count/(update_count+trans_dim_count),
            "Update Move Acceptance Probability" => update_acc_count / update_count,
            "Trans-Dimensional Move Acceptance Probability" => trans_dim_acc_count / trans_dim_count
        )
    output = SisMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 