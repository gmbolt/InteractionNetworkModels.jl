using Distances

export GibbsRandMove, GibbsScanMove

struct GibbsRandMove <: McmcMove
    ν::Int 
    counts::Vector{Int}
    function GibbsRandMove(;ν::Int=1)
        new(ν, [0,0])
    end 
end 

struct GibbsScanMove <: McmcMove
    ν::Int 
    counts::Vector{Int}
    function GibbsScanMove(;ν::Int=1)
        new(ν, [0,0])
    end 
end 

Base.show(io::IO, x::GibbsRandMove) = print(io, "GibbsRandMove(ν=$(x.ν))")
Base.show(io::IO, x::GibbsScanMove) = print(io, "GibbsScanMove(ν=$(x.ν))")


function accept_reject_entry!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    i::Int, 
    move::Union{GibbsRandMove,GibbsScanMove},
    model::VecMultigraphSNF
    ) 

    x_mode = model.mode
    # Proposal generation 
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:move.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    # Eval acceptance probability
    γ, d = (model.γ, model.d)
    log_α = γ * (d(x_curr, x_mode) - d(x_prop, x_mode))

    if log(rand()) < log_α
        @inbounds x_curr[i] = w_prop
        move.counts[1] += 1
    else 
        @inbounds x_prop[i] = w_curr
    end 
end 


# Optimised version for the hamming distances between mutlisets
function accept_reject_entry!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    i::Int, 
    move::Union{GibbsRandMove,GibbsScanMove},
    model::VecMultigraphSNF{Cityblock},
    ) 
    x_mode = model.mode
    # Proposal generation 
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:move.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    # Eval acceptance probability
    γ, d = (model.γ, model.d)
    @inbounds w_mode = x_mode[i]
    log_α = γ * (abs(w_curr - w_mode) - abs(w_prop - w_mode))
    
    if log(rand()) < log_α
        @inbounds x_curr[i] = w_prop
        move.counts[1] += 1
    else 
        @inbounds x_prop[i] = w_curr
    end 
end 

function accept_reject!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    move::GibbsRandMove, 
    model::VecMultigraphSNF
    ) 
    move.counts[2] += 1
    M = length(x_curr)
    i = rand(1:M)
    accept_reject_entry!(
        x_curr, x_prop, 
        i,
        move, model
    )
end 

function accept_reject!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    move::GibbsScanMove, 
    model::VecMultigraphSNF
    ) 
    move.counts[2] += length(x_curr)
    for i in eachindex(x_curr)
        accept_reject_entry!(
            x_curr, x_prop, 
            i,
            move, model
        )
    end 
end 

# A stripped-back proposal generation function (used in posterio samplers)
function prop_sample!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    move::GibbsRandMove
    )

    i = rand(1:length(x_curr))
    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:move.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    return 0.0
end 

function prop_sample_entry!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    move::GibbsScanMove,
    i::Int
    )

    @inbounds w_curr = x_curr[i]
    d = 1-2rand(0:1)
    w_tmp = w_curr + d * rand(1:move.ν)   
    w_prop = w_tmp ≥ 0 ? w_tmp : -w_tmp
    @inbounds x_prop[i] = w_prop

    return 0.0
end 