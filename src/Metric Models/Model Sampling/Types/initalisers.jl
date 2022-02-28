export McmcInitialiser, InitMode, InitRandEdit, InitRandIns, get_init
export perturb, perturb_ins

# Initilisers 
# -----------

# These can be passed to mcmc samplers to determine the defualt initialisation scheme. 

""" 
Abstract type representing initialisation schemes for SIS model samplers. 
"""
abstract type McmcInitialiser end


"""
`InitMode <: McmcInitialiser` - this is a MCMC initialisation scheme for SIS model samplers which starts the MCMC chain at the model mode by default.
"""
struct InitMode <: McmcInitialiser
    function InitMode()
        return new() 
    end 
end 

function get_init(
    initiliaser::InitMode,
    model::Union{SIS,SIM}, 
    )
    return model.mode
end 

struct InitRandEdit <: McmcInitialiser
    δ::Int
    function InitRandEdit(δ::Int) 
        return new(δ)
    end 
end 

function perturb(
    S::InteractionSequence,
    V::AbstractArray,
    K_inner::DimensionRange, 
    δ::Int
    )

    S_init = deepcopy(S)
    N = length(S)

    ind_del = zeros(Int, δ)
    ind_add = zeros(Int, δ)
    vals = zeros(Int, δ)

    rem_edits = δ

    for i in 1:N 
        if i == N 
            δ_tmp = rem_edits
        else 
            p = 1/(N-i+1)
            δ_tmp = rand(Binomial(rem_edits, p))
        end 

        if δ_tmp == 0
            continue 
        else

            
            n = length(S[i])
            d = rand(max(0, ceil(Int, (n + δ_tmp + - K_inner.u)/2)):min(n-K_inner.l, δ_tmp))
            m = n + δ_tmp - 2*d

            ind_del_v = view(ind_del, 1:d)
            ind_add_v = view(ind_add, 1:(δ_tmp-d))
            vals_v = view(vals, 1:(δ_tmp-d))

            StatsBase.seqsample_a!(1:n, ind_del_v)
            StatsBase.seqsample_a!(1:m, ind_add_v)
            sample!(V, vals)

            delete_insert!(S_init[i], ind_del_v, ind_add_v, vals_v)

        end 

        rem_edits -= δ_tmp 

        if rem_edits == 0 
            break 
        end 
    end 

    return S_init

end 



function get_init(
    initialiser::InitRandEdit,
    model::Union{SIS,SIM}
    )
    return perturb(model.mode, model.V, model.K_inner, initialiser.δ)
end 

struct InitRandIns <: McmcInitialiser
    δ::Int
    function InitRandIns(δ::Int) 
        return new(δ)
    end 
end 


function perturb_ins(
    S::InteractionSequence,
    V::AbstractArray,
    K_inner::DimensionRange, 
    δ::Int
    )

    S_init = deepcopy(S)
    N = length(S)

    ind_add = zeros(Int, δ)

    rem_edits = δ

    for i in 1:N 
        if i == N 
            δ_tmp = rem_edits
        else 
            p = 1/(N-i+1)
            δ_tmp = rand(Binomial(rem_edits, p))
        end 

        if δ_tmp == 0
            continue 
        else
            
            n = length(S[i])
            m = n + δ_tmp 
            if m > K_inner.u
                δ_tmp = (K_inner.u-n)
                m = n + δ_tmp
            end 
            ind_add_v = view(ind_add, 1:δ_tmp)
            StatsBase.seqsample_a!(1:m, ind_add_v)
            
            for j in ind_add_v 
                insert!(S_init[i], j, rand(V))
            end 

        end 

        rem_edits -= δ_tmp 

        if rem_edits == 0 
            break 
        end 
    end 

    return S_init

end 

function get_init(
    initialiser::InitRandIns,
    model::Union{SIS,SIM}
    )
    return perturb_ins(model.mode, model.V, model.K_inner, initialiser.δ)
end 

