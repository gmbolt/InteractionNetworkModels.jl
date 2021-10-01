using InvertedIndices, StatsBase

export imcmc_prop_sample_edit, imcmc_prop_sample_edit_informed
export imcmc_prop_sample_edit!, accept_reject_edit!, delete_insert!


function delete_insert!(
    x::Path, 
    δ::Int, d::Int,
    ind_del::AbstractArray{Int}, 
    ind_add::AbstractArray{Int}, 
    vals::AbstractArray{Int})

    @views for (i, index) in enumerate(ind_del[1:d])
        deleteat!(x, index - i + 1)
    end 
    @views for (index, val) in zip(ind_add[1:(δ-d)], vals[1:(δ-d)])
        # @show i, index, val
        insert!(x, index, val)
    end 

end 

# Alternative which just ennacts via the vectors given *no indexing)
function delete_insert!(
    x::Path, 
    ind_del::AbstractArray{Int}, 
    ind_add::AbstractArray{Int}, 
    vals::AbstractArray{Int})

    @views for (i, index) in enumerate(ind_del)
        deleteat!(x, index - i + 1)
    end 
    @views for (index, val) in zip(ind_add, vals)
        # @show i, index, val
        insert!(x, index, val)
    end 

end 

function imcmc_prop_sample_edit!(
    I_curr::Path{T}, I_prop::Path{T},
    δ::Int, # Number of edits 
    d::Int, # Number of deletions (addititions = δ - d)
    vertex_set::Vector{T},
    ind_del::AbstractArray{Int}, ind_add::AbstractArray{Int}, vals::AbstractArray{T} # Storage
    ) where {T<:Union{Int, String}}

    n = length(I_curr)
    m = n + δ - 2*d

    # Sample indexing info and new entries (all in-place)
    @views StatsBase.seqsample_a!(1:n, ind_del[1:d])
    @views StatsBase.seqsample_a!(1:m, ind_add[1:(δ-d)])
    @views sample!(vertex_set, vals[1:(δ-d)])

    delete_insert!(I_prop, δ, d, ind_del, ind_add, vals)

    log_ratio = log(min(n-1, δ)+1) - log(min(m-1, δ)+1) + (δ - 2*d) * log(length(vertex_set))
    
    return log_ratio
end 

ceil(Int, 4/2)

function imcmc_prop_sample_edit_informed(
    I_curr::Path{T},
    δ::Int, # Number of edits 
    d::Int, # Number of deletions (addititions = δ - d)
    vertex_dist::DiscreteUnivariateDistribution
    ) where {T<:Union{Int, String}}

    n = length(I_curr)
    m = n + δ - 2*d
    ind_del = StatsBase.seqsample_a!(1:n, zeros(Int, d))
    ind_add = StatsBase.seqsample_a!(1:m, zeros(Int, δ - d))

    I_prop = Vector{T}(undef, m)
    I_prop[Not(ind_add)] = I_curr[Not(ind_del)]
    I_prop[ind_add] = rand(vertex_dist, δ-d)

    log_ratio = log(min(n-1, δ)+1) - log(min(m-1, δ)+1) 

    for i in ind_del 
        @inbounds log_ratio += logpdf(vertex_dist, I_curr[i])
    end 
    for i in ind_add
        @inbounds log_ratio += -logpdf(vertex_dist, I_prop[i])
    end 
    
    return I_prop, log_ratio
end 

function accept_reject_edit!(
    I_curr::Path{T}, I_prop::Path{T},
    model::SPF{T},
    ν::Int,
    ind_del::AbstractArray{Int}, ind_add::AbstractArray{Int}, vals::AbstractArray{T}
    ) where {T<:Union{Int,String}}

    n = length(I_curr)
    δ = rand(1:ν) # Num. edits
    d = rand(0:min(n-1,δ)) # Num. deletions
    m = n + δ - 2*d

    # Set-up views 
    ind_del = view(ind_del, 1:d)
    ind_add = view(ind_add, 1:(δ-d))
    vals = view(vals, 1:(δ-d))

    # Sample indexing info and new entries (all in-place)
    StatsBase.seqsample_a!(1:n, ind_del)
    StatsBase.seqsample_a!(1:m, ind_add)
    sample!(model.V, vals)

    delete_insert!(I_prop, δ, d, ind_del, ind_add, vals)

    log_ratio = log(min(n-1, δ)+1) - log(min(m-1, δ)+1) + (δ - 2*d) * log(length(model.V))

    log_α = -model.γ * (
        model.dist(I_prop, model.mode) 
        - model.dist(I_curr, model.mode)
    ) + log_ratio

    if log(rand()) < log_α
        copy!(I_curr, I_prop)
        return 1
    else
        copy!(I_prop, I_curr)
        return 0
    end 
end 

function (mcmc::SpfInvolutiveMcmcEdit)(
    model::SPF{T};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Path{T}=model.mode
    ) where {T<:Union{Int,String}}


    # Find required length of chain
    req_samples = burn_in + 1 + (desired_samples - 1) * lag
    sample = fill(T[], req_samples)

    I_curr = mcmc.curr
    I_prop = mcmc.prop
    
    copy!(I_curr, init) # Initialisation
    copy!(I_prop, I_curr)

    # ind_del_mem = zeros(Int, mcmc.ν)
    # ind_add_mem = zeros(Int, mcmc.ν)
    # vals_mem = rand(model.V, mcmc.ν)
    # dist_store1 = zeros(Int, model.K+1)
    # dist_store2 = zeros(Int, model.K+1)

    lb(n::Int, δ::Int) = max(0, ceil(Int, (n + δ - model.K)/2))
    ub(n::Int, δ::Int) = min(n-1, δ)

    count=0

    for i in 1:req_samples

        n = length(I_curr)
        δ = rand(1:mcmc.ν) # Num. edits
        a, b = (lb(n, δ), ub(n, δ))
        # @show δ, a, b
        d = rand(a:b) # Num. deletions
        m = n + δ - 2*d

        # Set-up views 
        ind_del = view(mcmc.ind_del, 1:d)
        ind_add = view(mcmc.ind_add, 1:(δ-d))
        vals = view(mcmc.vals, 1:(δ-d))

        # Sample indexing info and new entries (all in-place)
        StatsBase.seqsample_a!(1:n, ind_del)
        StatsBase.seqsample_a!(1:m, ind_add)
        sample!(model.V, vals)

        # Enact delete insertion of entries
        delete_insert!(I_prop, δ, d, ind_del, ind_add, vals)

        # Evaluate log ratio (iMCMC ratio of auxiliary distributions)
        log_ratio = (log(b - a + 1) - log(ub(m,δ) - lb(m,δ) + 1) 
            + (δ - 2*d) * log(length(model.V))
            )
        # @show I_prop, log_ratio
        # Evaluate log acceptance probabilility
        log_α = -model.γ * (
            model.dist(I_prop, model.mode) 
            - model.dist(I_curr, model.mode)
        ) + log_ratio
        
        if log(rand()) < log_α
            copy!(I_curr, I_prop)
            count += 1
        else
            copy!(I_prop, I_curr)
        end 

        sample[i] = copy(I_curr)
    end 
    output = SpfMcmcOutput(
        model,
        sample[(burn_in+1):lag:end], 
        count/req_samples
        )
    return output
end

function StatsBase.sample!(
    mcmc::SpfInvolutiveMcmcEdit,
    model::SPF{T},
    output::Vector{Path{T}};
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::Path{T}=model.mode
    ) where {T<:Union{Int,String}}

    # Find required length of chain
    req_samples = burn_in + 1 + (length(output) - 1) * lag
    I_curr = mcmc.curr
    I_prop = mcmc.prop
    
    copy!(I_curr, init) # Initialisation
    copy!(I_prop, I_curr)

    count=1

    for i in 1:req_samples

        n = length(I_curr)
        δ = rand(1:mcmc.ν) # Num. edits
        d = rand(0:min(n-1,δ)) # Num. deletions
        m = n + δ - 2*d

        # Set-up views 
        ind_del = view(mcmc.ind_del, 1:d)
        ind_add = view(mcmc.ind_add, 1:(δ-d))
        vals = view(mcmc.vals, 1:(δ-d))

        # Sample indexing info and new entries (all in-place)
        StatsBase.seqsample_a!(1:n, ind_del)
        StatsBase.seqsample_a!(1:m, ind_add)
        sample!(model.V, vals)

        # Enact delete insertion of entries
        delete_insert!(I_prop, δ, d, ind_del, ind_add, vals)

        # Evaluate log ratio (iMCMC ratio of auxiliary distributions)
        log_ratio = log(min(n-1, δ)+1) - log(min(m-1, δ)+1) + (δ - 2*d) * log(length(model.V))

        # Evaluate log acceptance probabilility
        log_α = -model.γ * (
            model.dist(I_prop, model.mode) 
            - model.dist(I_curr, model.mode)
        ) + log_ratio
        
        if log(rand()) < log_α
            copy!(I_curr, I_prop)
        else
            copy!(I_prop, I_curr)
        end 

        if (i > burn_in) & (((i-1) % lag)==0)
            output[count] = copy(I_curr)
            count += 1
        end 
    end 
end