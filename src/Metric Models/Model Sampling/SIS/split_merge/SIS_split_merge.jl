using Distributions

const show_info = true

function imcmc_trans_dim_accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int}, 
    model::SIS, 
    mcmc::SisMcmcSplitMerge
    )

    K_in_lb, K_in_ub, K_out_lb, K_out_ub = (model.K_inner.l, model.K_inner.u, model.K_outer.l, model.K_outer.u)
    ν_td, p_ins, η, V, ind_td = (mcmc.ν_td, mcmc.p_ins, mcmc.η, model.V, mcmc.ind_td)

    prop_pointers = mcmc.prop_pointers
    curr_pointers = mcmc.curr_pointers
    γ, mode, d = (model.γ, model.mode, model.dist)

    N = length(S_curr)

    log_ratio = 0.0
    if show_info
        println("Split-merge move (pre prop enacted)...")
        println("Curr: $(S_curr)")
        println("Prop: $(S_prop)")
    end 
    is_split = rand(Bernoulli(0.5))
    if is_split
        ub = min(ν_td, N)
        ε = rand(1:ub)
        M = N + ε
        # Catch invalid proposal (only need upper since only increasing)
        if M > K_out_ub
            return 0 
        end 
        ind = view(ind_td, 1:ε)
        StatsBase.seqsample_a!(1:N, ind)
        if show_info
            println("Splitting at ind: $(ind)")
        end 
        log_ratio = multiple_adj_split_noisy!(
            S_prop, 
            prop_pointers, 
            ind, 
            p_ins, 
            η, V,
            K_in_lb, K_in_ub
        )
        log_ratio += log(ub) - log(min(floor(Int, M/2), ν_td))
    else 
        ub = min(floor(Int, N/2), ν_td)
        ε = rand(1:ub)
        M = N - ε
        # Catch invalid proposals
        if M < K_out_lb 
            return 0
        end 
        ind = view(ind_td, 1:ε)
        StatsBase.seqsample_a!(1:M, ind)
        # @show ind_merge
        if show_info
            println("Merging at ind: $(ind)")
        end 
        log_ratio = multiple_adj_merge_noisy!(
            S_prop,
            prop_pointers, 
            ind, 
            p_ins, η, V,
            K_in_lb, K_in_ub
        )
        log_ratio += log(ub) - log(min(M, ν_td))
    end 

    log_α = - γ * (d(mode, S_prop) - d(mode, S_curr)) + log_ratio
    if show_info
        println("Split-merge move (after prop enacted)...")
        println("Curr: $(S_curr)")
        println("Prop: $(S_prop)")
        @show d(mode, S_prop), d(mode, S_curr), log_α, log_ratio
    end
    if log(rand()) < log_α
        # We accept (make S_curr into S_prop)
        # println("Accepting")
        S_curr = deepcopy(S_prop)
        if is_split 
            # Copy split to S_curr
            live_index = 0
            for i in ind
                copy!(S_curr[i+live_index], S_prop[i+live_index])
                I_tmp = popfirst!(curr_pointers)
                copy!(I_tmp, S_prop[i+live_index+1])
                insert!(S_curr, i+live_index+1, I_tmp)
                live_index += 1
            end 
        else 
            # Copy merge to S_curr
            for i in ind
                I_tmp = popat!(S_curr, i+1)
                pushfirst!(curr_pointers, I_tmp)
                copy!(S_curr[i], S_prop[i])
            end 
        end 
        return 1 
    else 
        # We rejec (make S_prop back into S_curr)
        if is_split 
            # Undo split 
            for i in ind
                I_tmp = popat!(S_prop, i+1)
                pushfirst!(prop_pointers, I_tmp)
                copy!(S_prop[i], S_curr[i])
            end 
        else 
            # Undo merge 
            live_index = 0 
            for i in ind
                copy!(S_prop[i+live_index], S_curr[i+live_index])
                I_tmp = popfirst!(prop_pointers)
                copy!(I_tmp, S_curr[i+live_index+1])
                insert!(S_prop, i+live_index+1, I_tmp)
                live_index += 1
            end
        end 
        return 0
    end 
end 

# Sampler Functions 
# -----------------
"""
    draw_sample!(
        sample_out::InteractionSequenceSample, 
        mcmc::SisMcmcSplitMerge, 
        model::SIS;
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::InteractionSequence=get_init(model, mcmc.init)
    )

Draw sample in-place from given SIS model `model::SIS` via MCMC algorithm with edit allocation and interaction insertion/deletion, storing output in `sample_out::InteractionSequenceSample`. 

Accepts keyword arguments to change MCMC output, including burn-in, lag and initial values. If not given, these are set to the default values of the passed MCMC sampler `mcmc::SisMcmcInsertDelete`.
"""
function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int}, SubArray},
    mcmc::SisMcmcSplitMerge,
    model::SIS;
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

"""
    draw_sample(
        mcmc::SisMcmcInsertDelete, 
        model::SIS;
        desired_samples::Int=mcmc.desired_samples, 
        burn_in::Int=mcmc.burn_in,
        lag::Int=mcmc.lag,
        init::Vector{Path{T}}=get_init(model, mcmc.init)
        )
"""
function draw_sample(
    mcmc::SisMcmcSplitMerge, 
    model::SIS;
    desired_samples::Int=mcmc.desired_samples, 
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
    ) 

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out

end 


function (mcmc::SisMcmcSplitMerge)(
    model::SIS;
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
    output = SisMcmcOutput(
            model, 
            sample_out, 
            p_measures
            )

    return output

end 