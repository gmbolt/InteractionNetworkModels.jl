using RecipesBase, Measures


@recipe function f(
    output::SisPosteriorMcmcOutput,
    S_true::InteractionSequence{Int}
    ) 

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

@recipe function f(
    outputs::Vector{SisPosteriorMcmcOutput},
    S_true::InteractionSequence{Int}
    ) 
    d = outputs[1].posterior.dist
    S_samples = [x.S_sample for x in outputs]
    γ_samples = [x.γ_sample for x in outputs]
    layout := (2,1)
    k = length(outputs)
    subplot := [i ≤ k ? 1 : 2 for i in 1:(2*k)]
    legend --> [true false]
    label --> ["Chain $i" for j in 1:1, i in ncycle(1:k, 2)]
    xguide --> "Index"
    yguide --> ["Distance from Reference" "γ"]
    size --> (800, 600)
    margin --> 5mm
    alpha --> 0.5
    y1 = [map(y->d(S_true,y), x) for x in S_samples]
    y2 = γ_samples 
    vcat(y1,y2)

    
end 


@recipe function f(output::SisPosteriorModeConditionalMcmcOutput, S_true::Vector{Path{Int}}) 
    S_sample = output.S_sample
    d = output.posterior.dist
    xguide --> "Index"
    yguide --> "Distance from Truth"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    map(x->d(S_true,x), S_sample)
end 

@recipe function f(output::SisPosteriorDispersionConditionalMcmcOutput) 
    xguide --> "Index"
    yguide --> "γ"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    output.γ_sample
end 


@recipe function f(
    output::SimPosteriorMcmcOutput,
    S_true::InteractionSequence{Int}
    ) 

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

@recipe function f(
    outputs::Vector{SimPosteriorMcmcOutput},
    S_true::InteractionSequence{Int}
    ) 
    d = outputs[1].posterior.dist
    S_samples = [x.S_sample for x in outputs]
    γ_samples = [x.γ_sample for x in outputs]
    layout := (2,1)
    k = length(outputs)
    subplot := [i ≤ k ? 1 : 2 for i in 1:(2*k)]
    legend --> [true false]
    label --> ["Chain $i" for j in 1:1, i in ncycle(1:k, 2)]
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Reference" "γ"]
    size --> (800, 600)
    margin --> 5mm
    alpha --> [i ≤ k ? 1.0 : 0.5 for j in 1:1, i in 1:(2*k)]
    y1 = [map(y->d(S_true,y), x) for x in S_samples]
    y2 = γ_samples 
    vcat(y1,y2)
end 

@recipe function f(output::SimPosteriorModeConditionalMcmcOutput, S_true::Vector{Path{Int}}) 
    S_sample = output.S_sample
    d = output.posterior.dist
    xguide --> "Index"
    yguide --> "Distance from Truth"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    map(x->d(S_true,x), S_sample)
end 

@recipe function f(output::SimPosteriorDispersionConditionalMcmcOutput) 
    xguide --> "Index"
    yguide --> "γ"
    size --> (800, 300)
    label --> nothing
    margin --> 5mm
    output.γ_sample
end 



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
    y1 = map(x -> output.posterior.dist(I_true, x), I_sample)
    y1
end 

@recipe function f(output::SpfPosteriorDispersionConditionalMcmcOutput{T}) where T <:Union{Int, String}
    xguide --> "Index"
    yguide --> "γ"
    legend --> false
    size --> (800, 600)
    output.γ_sample
end 

