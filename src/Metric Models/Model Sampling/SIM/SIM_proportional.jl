export imcmc_proportional_multinomial_edit_accept_reject!

function imcmc_multinomial_edit_accept_reject!(
    S_curr::InteractionSequence, 
    S_prop::InteractionSequence, 
    model::SIM, 
    mcmc::SimMcmcInsertDeleteProportional
    ) 

    N = length(S_curr)  
    n_tot = sum(length,S_curr)
    m_tot = n_tot
    K_in_lb = model.K_inner.l
    K_in_ub = model.K_inner.u
    δ = rand(1:mcmc.ν_ed)  # Number of edits to enact 
    rem_edits = δ # Remaining edits to allocate
    len_diffs = 0
    j = 0 # Keeps track how many interaction have been edited 
    Z = 1 # Normalising term 
    log_prod_term = 0.0 

    # println("Making $δ edits in total...")
    for (i,(I_curr,I_prop)) in enumerate(zip(S_curr, S_prop))
        # If at end we just assign all remaining edits to final interaction 
        n = length(I_curr)
        if i == N 
            δ_tmp = rem_edits
        # Otherwise we sample the number of edits via rescaled Binomial 
        else 
            log_p_tmp = log(n) - log(n_tot)
            p = exp(log_p_tmp - log(Z))
            δ_tmp = rand(Binomial(rem_edits, p)) # Number of edits to perform on ith interaction 
            Z -= exp(log_p_tmp)
        end 
        # println("   Index $i getting $δ_tmp edits...")
        # If we sampled zero edits we skip to next iteration 
        if δ_tmp == 0
            continue 
        else
            j += 1 # Increment j 
            # Make edits .... 
            # a,b = (lb(n, δ_tmp, model), ub(n, δ_tmp))
            d = rand(0:min(n,δ_tmp))
            m = n + δ_tmp - 2*d
            m_tot += δ_tmp - 2*d
            # Catch invalid proposals
            if (m < K_in_lb) | (m > K_in_ub)
                # Here we just reject the proposal
                for (I_prop, I_curr) in zip(S_prop,S_curr)
                    @inbounds copy!(I_prop, I_curr)
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

            @inbounds delete_insert!(I_prop, ind_del, ind_add, vals)

            @inbounds mcmc.ind_update[j] = i # Store which interaction was updated
            
            # Add to log_ratio
            # log_prod_term += log(b - a + 1) - log(ub(m, δ_tmp) - lb(m, δ_tmp, model) +1)
            log_prod_term += log(min(n, δ_tmp)+1) - log(min(m, δ_tmp)+1) + log(m) - log(n)
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
    log_ratio = log(length(model.V)) * len_diffs + log_prod_term + N*(log(n_tot) - log(m_tot))
    # log_ratio = log_dim_diff + log_prod_term
    log_lik_ratio = -model.γ * (
        model.dist(model.mode, S_prop)-model.dist(model.mode, S_curr)
        )

    log_multinom_ratio_term = log_multinomial_ratio(S_curr, S_prop)
    # @show log_dim_diff, log_prod_term, log_lik_ratio
        
    # Log acceptance probability
    log_α = log_lik_ratio + log_ratio + log_multinom_ratio_term

    # Accept-reject step. Use info in mcmc.ind_update to know which interaction are to be copied over 
    if log(rand()) < log_α
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_curr[i], S_prop[i])
        end
        return 1 
    else 
        for i in view(mcmc.ind_update, 1:j)
            @inbounds copy!(S_prop[i], S_curr[i])
        end 
        return 0 
    end 
end 


# Sampler Functions 
# -----------------
"""
    draw_sample!(
        sample_out::InteractionSequenceSample, 
        mcmc::SimMcmcInsertDeleteProportional, 
        model::SIM;
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::InteractionSequence=get_init(model, mcmc.init)
    )

Draw sample in-place from given SIM model `model::SIM` via MCMC algorithm with edit allocation and interaction insertion/deletion, storing output in `sample_out::InteractionSequenceSample`. 

Accepts keyword arguments to change MCMC output, including burn-in, lag and initial values. If not given, these are set to the default values of the passed MCMC sampler `mcmc::SimMcmcInsertDelete`.
"""
function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SimMcmcInsertDeleteProportional,
    model::SIM;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

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
            @inbounds sample_out[sample_count] = deepcopy(S_curr)
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
    mcmc::SimMcmcInsertDeleteProportional, 
    model::SIM;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 

function (mcmc::SimMcmcInsertDeleteProportional)(
    model::SIM;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
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
    output = SimMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 