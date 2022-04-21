using RecipesBase
export McmcSampler, SnfMcmcOutput

struct McmcSampler{T<:McmcMove}
    move::T
    output::McmcOutputParameters
    function McmcSampler(
        move::T;
        desired_samples::Int=1000, burn_in::Int=0, lag::Int=1
        ) where {T<:McmcMove}
        output = McmcOutputParameters(desired_samples, burn_in, lag)
        new{T}(move, output)
    end 
end 

Base.show(io::IO, x::McmcSampler{T}) where {T} = print(io, typeof(x))

acceptance_prob(mcmc::McmcSampler) = acceptance_prob(mcmc.move)


struct SnfMcmcOutput{T,N,S}
    sample::Vector{Array{T,N}}
    model::SNF{T,N,S}
end 

Base.show(io::IO, x::SnfMcmcOutput) = print(io, typeof(x))

@recipe function f(output::SnfMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.d(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end 


function draw_sample!(
    sample_out::Union{Vector{Vector{Int}},SubArray},
    mcmc::McmcSampler, 
    model::VecMultigraphSNF;
    burn_in::Int=mcmc.output.burn_in,
    lag::Int=mcmc.output.lag,
    init::Vector{Int}=model.mode
    )

    x_curr = copy(init)
    x_prop = copy(x_curr)

    sample_count = 1 
    i = 0
    reset_counts!(mcmc.move)

    while sample_count ≤ length(sample_out)
        i += 1 
        # Store value 
        if (i > burn_in) & (((i-1) % lag)==0)
            @inbounds sample_out[sample_count] = deepcopy(x_curr)
            sample_count += 1
        end 
        accept_reject!(
            x_curr, x_prop, 
            mcmc.move,
            model
        )
    end 

end 


function draw_sample(
    mcmc::McmcSampler, 
    model::VecMultigraphSNF;
    desired_samples::Int=mcmc.output.desired_samples, 
    args...
    ) 

    sample_out = Vector{Vector{Int}}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model; args...)
    return sample_out
end 

function (mcmc::McmcSampler)(
    model::VecMultigraphSNF;
    args...
    )

    sample_out = draw_sample(mcmc, model; args...)

    return SnfMcmcOutput(sample_out, model)

end 