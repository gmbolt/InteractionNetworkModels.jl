using Distributions

export imcmc_noisy_split, complement_rand
export split_noisy!, split_noisy, rand_split_noisy!, rand_split_noisy
export merge_noisy!, merge_noisy, merge_noisy_ins_dels!, rand_merge_noisy_ins_dels!
export multiple_adj_split_noisy!, multiple_adj_merge_noisy!
export rand_multiple_adj_split_noisy!, rand_multiple_adj_merge_noisy!

function complement_rand(
    V::UnitRange,
    x::Int
    )
    tmp = rand(V[1:(end-1)])
    if tmp ≥ x
        return tmp+1
    else 
        return tmp
    end  
end 

# ===============================================
# Functions to split/merge two paths (with noise)
# ===============================================

# Splitting functions 
# -------------------
function split_noisy!(
    curr::Path,  # Current path 
    store::Path, # Storage for new path
    ind_del::AbstractArray{Int}, 
    ind1_add::AbstractArray{Int}, 
    ind2_add::AbstractArray{Int}, 
    vals1::AbstractArray{Int},
    vals2::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )
    copy!(store, curr)
    prop1, prop2 = (curr, store) # Aliases (more readable)
    # @show prop1, prop2
    @views for (i, index) in enumerate(ind_del)
        deleteat!(prop1, index - i + 1)
        deleteat!(prop2, index - i + 1)
    end 
    # @show x,y
    n_diff = 0
    for (i, val) in enumerate(prop1)
        if rand() > η
            n_diff += 1
            if rand() < 0.5 
                # println("flip in first")
                prop1[i] = complement_rand(V, val)
            else 
                # println("flip in second")
                prop2[i] = complement_rand(V, val)
            end 
        end 

    end 

    @views for (index, val) in zip(ind1_add, vals1)
        # @show i, index, val
        insert!(prop1, index, val)
    end 
    @views for (index, val) in zip(ind2_add, vals2)
        # @show i, index, val
        insert!(prop2, index, val)
    end 
    return n_diff
end 

function split_noisy!(
    curr::Path,
    prop1::Path, prop2::Path,
    ind_del::AbstractArray{Int}, 
    ind1_add::AbstractArray{Int}, 
    ind2_add::AbstractArray{Int}, 
    vals1::AbstractArray{Int},
    vals2::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )
    copy!(prop1, curr)

    n_diff = split_noisy!(
        prop1, prop2, 
        ind_del, ind1_add, ind2_add, 
        vals1, vals2, 
        η, V
    )

    return n_diff
end 


function split_noisy(
    curr::Path, 
    ind_del::AbstractArray{Int}, 
    ind_add_x::AbstractArray{Int}, 
    ind_add_y::AbstractArray{Int}, 
    vals_x::AbstractArray{Int},
    vals_y::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )
    prop1 = copy(curr)
    prop2 = copy(curr)
    split_noisy!(
        curr,
        prop1, prop2,
        ind_del, 
        ind_add_x, ind_add_y, 
        vals_x, vals_y, η, V
        )
    return prop1, prop2
end 

# Noisy split move, here η denotes the probability of entry flips 


function rand_split_noisy!(
    curr::Path{Int}, # Current path
    store::Path{Int}, # Storage for new path 
    p_del::TrGeometric, # Distribution on deleltions 
    p_ins::Geometric, # Distribution on additions
    η::Float64, # Noise parameters 
    V::UnitRange, # Vertex set 
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )

    # Sample auxiliary info
    n = length(curr)
    d = rand(p_del) # deleltions
    a1, a2 = rand(p_ins,2) # Insertions 
    m1, m2 = n - d .+ (a1, a2)
    # Indexing info 
    ind_del = zeros(Int, d) 
    ind1_add = zeros(Int, a1)
    ind2_add = zeros(Int, a2)
    StatsBase.seqsample_a!(1:n, ind_del)
    StatsBase.seqsample_a!(1:m1, ind1_add)
    StatsBase.seqsample_a!(1:m2, ind2_add)

    vals1 = rand(V, a1) 
    vals2 = rand(V, a2)
    # @show d, a1, a2
    # @show ind_del, ind1_add, ind2_add, vals1, vals2
    n_diff = split_noisy!(
        curr,
        store,
        ind_del,
        ind1_add, ind2_add,
        vals1, vals2, 
        η, 
        V
    )
    p = p_del.p
    p_del_rev = TrGeometric(p, min(m1,m2))
    k = n - d  # Number of entries kept
    # @show n_diff, k, m1, m2
    log_ratio = (
        log(normalising_const(p_del_rev)) - log(normalising_const(p_del))
        + (k - abs(m1-m2)) * log(1-p) - log(p)
        + n_diff * log(length(V) - 1) 
        - (k - n_diff) * log(η) - n_diff * log(1 - η) 
        + (m1 + m2 - n - k) * log(length(V))
    )
    # Here we catch invalid proposals
    if (m1 < K_in_lb) | (m1 > K_in_ub) | (m2 < K_in_lb) | (m2 > K_in_ub)
        log_ratio += -Inf
    end 

    return log_ratio
end


function rand_split_noisy!(
    curr::Path{Int}, # Current path
    prop1::Path{Int}, prop2::Path{Int}, # Two new paths
    p_del::TrGeometric, # Distribution on deleltions 
    p_ins::Geometric, # Distribution on additions
    η::Float64, # Noise parameters 
    V::UnitRange, # Vertex set 
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )

    copy!(prop1, curr)
    log_ratio = rand_split_noisy!(
        prop1, prop2, 
        p_del, p_ins, 
        η, V,
        K_in_lb, K_in_lb
    )
    return log_ratio
end

function rand_split_noisy(
    curr::Path{Int}, # Current path
    p_del::TrGeometric, # Distribution on deleltions 
    p_ins::Geometric, # Distribution on additions
    η::Float64, # Noise parameters 
    V::UnitRange, # Vertex set 
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )

    prop1 = Int[]
    prop2 = Int[]

    log_ratio = rand_split_noisy!(
        curr, 
        prop1, prop2, 
        p_del, p_ins,
        η, 
        V,
        K_in_lb, K_in_ub
    )

    return prop1, prop2, log_ratio

end 

# Merging functions
# -----------------

function merge_noisy!(
    curr1::Path, curr2::Path, 
    prop::Path, 
    ind1_keep::AbstractArray{Int},
    ind2_keep::AbstractArray{Int},
    ind_add::AbstractArray{Int},
    vals::AbstractArray{Int}
    )

    # NOTE we must have that deletions applied to curr1 and curr2 lead to subsequences of the same length. 
    @assert length(prop) == length(ind1_keep) "Invalid length of input `prop`"
    for i in eachindex(prop)
        if rand() < 0.5 
            # println("first")
            prop[i] = curr1[ind1_keep[i]]
        else 
            # println("second")
            prop[i] = curr2[ind2_keep[i]]
        end 
    end 
    # @show prop
    @views for (index, val) in zip(ind_add, vals)
        # @show i, index, val
        insert!(prop, index, val)
    end 
end 

function merge_noisy_ins_dels!(
    curr1::Path, 
    curr2::Path, 
    ind1_del::AbstractArray{Int},
    ind2_del::AbstractArray{Int},
    ind_add::AbstractArray{Int},
    vals::AbstractArray{Int}
    )

    @assert (length(curr1)-length(ind1_del)) == (length(curr2)-length(ind2_del)) "Invalid delition plan."

    @views for (i, index) in enumerate(ind1_del)
        deleteat!(curr1, index - i + 1)
    end 
    @views for (i, index) in enumerate(ind2_del)
        deleteat!(curr2, index - i + 1)
    end 
    
    # Competition over entries...
    # We treat curr1 and the desired output, deleting all of curr2
    n_diff = 0
    for i in eachindex(curr1)
        # W.p. we set the ith entry ot curr1 to the corresponding entry of curr2 
        if curr1[i] !== curr2[1]
            n_diff += 1
            if rand() < 0.5 
                # println("second")
                curr1[i] = popfirst!(curr2) # Note this will return first val of curr2 and remove it 
            else
                # println("first")
                popfirst!(curr2) # Here we just remove the first entry of curr2, not assigning this to curr1 
            end 
        else 
            popfirst!(curr2) 
        end 
    end 

    # Insertions....
    @views for (index, val) in zip(ind_add, vals)
        # @show i, index, val
        insert!(curr1, index, val)
    end 

    return n_diff
end 

function merge_noisy(
    curr1::Path, 
    curr2::Path, 
    ind1_keep::AbstractArray{Int},
    ind2_keep::AbstractArray{Int},
    ind_add::AbstractArray{Int},
    vals::AbstractArray{Int}
    )
    prop = zeros(Int, length(ind1_keep)) 
    # @show prop
    merge_noisy!(
        curr1, curr2,
        prop, 
        ind1_keep, ind2_keep, 
        ind_add, 
        vals 
    )
    return prop
end 

function rand_merge_noisy_ins_dels!(
    curr1::Path{Int}, 
    curr2::Path{Int}, 
    p_del::TrGeometric,
    p_ins::Geometric, 
    η::Float64,
    V::AbstractArray{Int},
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )
    n1, n2 = (length(curr1), length(curr2))
    # Sample auxiliary info
    k = min(n1, n2) - rand(p_del)
    a = rand(p_ins)
    # Indexing subseqs (these cant be fixed in memory since possible unbounded)
    ind1_del = zeros(Int, n1-k)
    ind2_del = zeros(Int, n2-k)
    ind_add = zeros(Int, a)
    StatsBase.seqsample_a!(1:n1, ind1_del)
    StatsBase.seqsample_a!(1:n2, ind2_del)
    StatsBase.seqsample_a!(1:(k+a), ind_add)
    # New entries 
    vals = rand(V, a)
    n_diff = merge_noisy_ins_dels!(
        curr1, curr2, 
        ind1_del, ind2_del, 
        ind_add, 
        vals
    )
    # Evaluate log ratio 
    p = p_del.p
    m = k + a
    p_rev = TrGeometric(p, m)

    log_ratio = (
        log(normalising_const(p_rev)) - log(normalising_const(p_del))
        + (k - n_diff) * log(η) + n_diff * log(1 - η) 
        - n_diff * log(length(V) - 1) 
        + (k + m - n1 - n2) * log(length(V))
    )
    # Here we catch invalid proposals
    if (m < K_in_lb) | (m > K_in_ub) 
        log_ratio += -Inf
    end 
    return log_ratio
end 


# =======================================
# Functions to split/merge multiple paths
# =======================================

# With subsequence given 
# ----------------------

# This will split adjacent paths 
function multiple_adj_split_noisy!(
    curr::Vector{Path{Int}}, 
    store::Vector{Path{Int}},  # Storage for new paths 
    ind_split::AbstractArray{Int}, # Which to split 
    p_ins::Geometric,
    η::Float64,
    V::UnitRange,
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )
    p = p_ins.p
    live_index = 0 # We need to keep adjusting indexing as new paths are interoduced    for i in ind_split 
    log_ratio = 0.0
    for i in ind_split
        I_curr = curr[i+live_index] # Current path 
        p_del = TrGeometric(p, length(I_curr)) # Deletion distribution 
        I_new = popfirst!(store)  # Storage for new path 
        insert!(curr, i+1+live_index, I_new)
        live_index += 1
        log_ratio += rand_split_noisy!(
            I_curr, I_new,
            p_del, p_ins,
            η, 
            V,
            K_in_lb, K_in_ub
        )
    end 
    return log_ratio
end 

function multiple_adj_merge_noisy!(
    S_curr::Vector{Path{Int}},
    S_store::Vector{Path{Int}}, 
    ind_merge::AbstractArray{Int}, #This is the index where merged path will go
    p_ins::Geometric, 
    η::Float64, V::UnitRange,
    K_in_lb::Real, K_in_ub::Real # Inner dimension bounds
    )
    p = p_ins.p
    log_ratio = 0.0
    for i in ind_merge
        curr1, curr2 = S_curr[i:(i+1)]
        p_del = TrGeometric(p, min(length(curr1), length(curr2)))
        log_ratio += rand_merge_noisy_ins_dels!(
            curr1, curr2,
            p_del, p_ins, 
            η, V,
            K_in_lb, K_in_ub
        )
        I_old = popat!(S_curr, i+1)
        pushfirst!(S_store, I_old)
    end  
    return log_ratio
end 

# With subsequence random
# --------------------------

function rand_multiple_adj_split_noisy!(
    S_curr::InteractionSequence{Int},
    S_store::Vector{Path{Int}},
    ν_td::Int,
    p_ins::Geometric, 
    η::Float64, 
    V::UnitRange,
    K_in_lb::Real, K_in_ub::Real, # Inner dimension bounds
    K_out_ub::Real, # Outer dimension upper bound,
    ind_td::AbstractArray{Int}
    )

    N = length(S_curr)
    ub = min(ν_td, N)
    ε = rand(1:ub)
    M = N + ε
    # Catch invalid proposal (only need upper since only increasing)
    if M > K_out_ub
        return -Inf 
    end 
    ind_split = view(ind_td, 1:ε)
    StatsBase.seqsample_a!(1:N, ind_split)
    # @show ind_split
    log_ratio = multiple_adj_split_noisy!(
        S_curr, 
        S_store, 
        ind_split, 
        p_ins, 
        η, V,
        K_in_lb, K_in_ub
    )
    log_ratio += log(ub) - log(min(floor(Int, M/2), ν_td))

    return log_ratio, ε
end 

function rand_multiple_adj_split_noisy!(
    S_curr::InteractionSequence{Int},
    S_store::Vector{Path{Int}},
    model::SIS,
    mcmc::SisMcmcSplitMerge
    )

    K_in_lb, K_in_ub, K_out_ub = (model.K_inner.l, model.K_inner.u, model.K_outer.u)
    ν_td, p_ins, η, V, ind_td = (mcmc.ν_td, mcmc.p_ins, mcmc.η, model.V, mcmc.ind_td)

    return rand_multiple_adj_split_noisy!(
        S_curr, S_store, 
        ν_td, p_ins, η, V,
        K_in_lb, K_in_ub, K_out_ub,
        ind_td
    )
end 

function rand_multiple_adj_merge_noisy!(
    S_curr::InteractionSequence{Int},
    S_store::Vector{Path{Int}},
    ν_td::Int, 
    p_ins::Geometric, 
    η::Float64, 
    V::UnitRange,
    K_in_lb::Real, K_in_ub::Real, # Inner dimension bounds
    K_out_lb::Real, # Outer dimension lower bound
    ind_td::AbstractArray{Int}
    )

    N = length(S_curr)
    ub = min(floor(Int, N/2), ν_td)
    ε = rand(1:ub)
    M = N - ε
    # Catch invalid proposals
    if M < K_out_lb 
        return -Inf 
    end 
    ind_merge = view(ind_td, 1:ε)
    StatsBase.seqsample_a!(1:M, ind_merge)
    # @show ind_merge
    log_ratio = multiple_adj_merge_noisy!(
        S_curr,
        S_store, 
        ind_merge, 
        p_ins, η, V,
        K_in_lb, K_in_ub
    )

    log_ratio += log(ub) - log(min(M, ν_td))

    return log_ratio, ε
end 

function rand_multiple_adj_merge_noisy!(
    S_curr::InteractionSequence{Int},
    S_store::Vector{Path{Int}},
    model::SIS,
    mcmc::SisMcmcSplitMerge
    )

    K_in_lb, K_in_ub, K_out_lb = (model.K_inner.l, model.K_inner.u, model.K_outer.l)
    ν_td, p_ins, η, V, ind_td = (mcmc.ν_td, mcmc.p_ins, mcmc.η, model.V, mcmc.ind_td)

    return rand_multiple_adj_merge_noisy!(
        S_curr, S_store, 
        ν_td, p_ins, η, V,
        K_in_lb, K_in_ub, K_out_lb,
        ind_td
    )
end 