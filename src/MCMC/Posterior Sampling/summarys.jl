using RecipesBase, Measures

# =======
#   SIS
# =======

@recipe function f(
    output::SisPosteriorMcmcOutput{T},
    S_true::InteractionSequence{T}
    ) where {T<:Union{Int, String}}

    S_sample = output.S_sample
    γ_sample = output.γ_sample
    layout := (2,1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "γ"]
    size --> (800, 600)
    margin --> 5mm
    y1 = map(x->output.posterior.dist(S_true,x), S_sample)
    y2 = γ_sample 
    hcat(y1,y2)
end 


@recipe function f(output::SisPosteriorModeConditionalMcmcOutput{T}, S_true::Vector{Path{T}}) where {T<:Union{Int, String}}
    S_sample = output.S_sample
    xguide --> "Index"
    yguide --> "Distance from Truth"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    map(x->output.dist(S_true,x), S_sample)
end 

@recipe function f(output::SisPosteriorDispersionConditionalMcmcOutput{T}) where {T<:Union{Int,String}}
    xguide --> "Index"
    yguide --> "γ"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    output.γ_sample
end 


# ========
#   SIM 
# ========

@recipe function f(
    output::SimPosteriorMcmcOutput{T},
    S_true::InteractionSequence{T}
    ) where {T<:Union{Int, String}}

    S_sample = output.S_sample
    γ_sample = output.γ_sample
    layout := (2,1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "γ"]
    size --> (800, 600)
    margin --> 5mm
    y1 = map(x->output.posterior.dist(S_true,x), S_sample)
    y2 = γ_sample 
    hcat(y1,y2)
end 


@recipe function f(output::SimPosteriorModeConditionalMcmcOutput{T}, S_true::Vector{Path{T}}) where {T<:Union{Int, String}}
    S_sample = output.S_sample
    xguide --> "Index"
    yguide --> "Distance from Truth"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    map(x->output.dist(S_true,x), S_sample)
end 

@recipe function f(output::SimPosteriorDispersionConditionalMcmcOutput{T}) where {T<:Union{Int,String}}
    xguide --> "Index"
    yguide --> "γ"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    output.γ_sample
end 


# ========
#   SPF
# ========


@recipe function f(output::T) where {T<:SpfPosteriorMcmcOutput}
    xguide --> "Index"
    yguide --> "γ"
    legend --> false
    size --> (800, 300)
    label := nothing
    output.γ_sample
end 


@recipe function f(output::SpfPosteriorMcmcOutput{T}, I_true::Path{T}) where T <:Union{Int, String}
    I_sample = output.I_sample
    xguide --> "Index"
    yguide --> ["Distance from Truth" "γ"]
    legend --> false
    size --> (800, 600)
    label := [nothing nothing]
    layout := (2,1)
    y1 = map(x -> output.dist(I_true, x), I_sample)
    y2 = output.γ_sample
    hcat(y1, y2)
end 

@recipe function f(output::SpfPosteriorModeConditionalMcmcOutput{T}, I_true::Path{T}) where T <:Union{Int, String}
    I_sample = output.I_sample
    xguide --> "Index"
    yguide --> "Distance from Truth"
    legend --> false
    size --> (800, 600)
    y1 = map(x -> output.dist(I_true, x), I_sample)
    y1
end 

@recipe function f(output::SpfPosteriorDispersionConditionalMcmcOutput{T}) where T <:Union{Int, String}
    xguide --> "Index"
    yguide --> "γ"
    legend --> false
    size --> (800, 600)
    output.γ_sample
end 

