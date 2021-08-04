using RecipesBase, Distances, Multisets, StatsBase
export get_dist_dict, KL_error, SummaryPlot, get_inner_len_dist, get_outer_len_dist

# # Plot Recipes 
@recipe function f(output::SisMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 400)
    x
end 

@recipe function f(output::SimMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 400)
    x
end 

@recipe function f(output::SpfMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 400)
    x
end 

@userplot LengthPlot

@recipe function f(h::LengthPlot)
    input = h.args
    @assert length(input) == 1 "Accepts only a single argument"
    @assert typeof(input[1]) == SpfMcmcOutput "Input must be of type SpfMcmcOutput"
    size --> (800, 400)
    legend --> false
    data = length.(input[1].sample)
    @series begin
        seriestype := :bar
        yguide --> "Proportion"
        xguide --> "Path Length"
        xticks --> 1:maximum(data)
        xlims --> [0, maximum(data)+0.5]
        proportionmap(data)
    end 

end 


@userplot SummaryPlot
@recipe function f(h::SummaryPlot)
    output = h.args  # A mcmc_output type
    model = output[1].model
    sample = output[1].sample
    layout := (3,1)
    legend --> false
    xguide := "Sample"
    size --> (800, 600)

    @series begin
        seriestype := :line
        yguide --> "Distance from Mode"
        map(x -> model.dist(model.mode, x), sample)
    end 
    @series begin
        seriestype := :line
        yguide --> "Num. Interactions"
        map(length, sample)
    end 
    @series begin
        seriestype := :line
        yguide --> "Mean Interaction Length"
        map(x -> mean(length.(x)), sample)
    end 
end 

# Get Normalising Constant for path model

# Return approximate distribution over 

function get_dist_dict(output::SisMcmcOutput)
    d = Dict{Vector{Path}, Real}()
    N = length(output.sample)

    for p in output.sample
        myaddcounts!(d, p)
    end 

    # normalise
    map!(x -> x/N, values(d))
    return d
end 

function get_dist_dict(output::SimMcmcOutput)
    d = Dict{Multiset{Path}, Real}()
    N = length(output.sample)
    for p in output.sample
        myaddcounts!(d, Multiset(p))
    end 
    # normalise
    map!(x -> x/N, values(d))
    return d
end 

"""
`KL_error(output::SisMcmcOutput)`

Find the KL-divergence between the true distribution and the approximation obtained via
the MCMC sample `output`. 

That is, if ``π(\\mathcal{S})`` is the true distribution and ``\\tilde{\\pi}(\\mathcal{S})`` is the approximation then this function will return 

```math
\\sum_{\\mathcal{S}} \\tilde{\\pi}(\\mathcal{S}) \\log\\left( \\frac{\\tilde{\\pi}(\\mathcal{S})}{\\pi(\\mathcal{S})}\\right)
```

where we assume that ``0 \\cdot \\log(0) = 0``.  
"""
function KL_error(output::SisMcmcOutput)
    d_approx = get_dist_dict(output)
    d_true = get_true_dist_dict(output.model)
    D = 0.0
    for key in keys(d_approx)
        D += d_approx[key] * ( log(d_approx[key]) - log(d_true[key]) )
    end 
    return D
end 

"""
`KL_error(output::SisMcmcOutput, true_dist::Dict)`

Takes the true distribution, as represented by a dictionary, as an additional parameter.
"""
function KL_error(output::SisMcmcOutput, true_dist::Dict)
    d_approx = get_dist_dict(output)
    D = 0.0
    for key in keys(d_approx)
        D += d_approx[key] * ( log(d_approx[key]) - log(true_dist[key]) )
    end 
    return D
end 

"""
`KL_error(output::SimMcmcOutput, true_dist::Dict)`

Takes the true distribution, as represented by a dictionary, as an additional parameter.
"""
function KL_error(output::SimMcmcOutput, true_dist::Dict)
    d_approx = get_dist_dict(output)
    D = 0.0
    for key in keys(d_approx)
        D += d_approx[key] * ( log(d_approx[key]) - log(true_dist[key]) )
    end 
    return D
end 


function get_outer_len_dist(mcmc_output::SisMcmcOutput, α::Float64)
    p = proportions(length.(mcmc_output.sample), 1:mcmc_output.model.K_outer)
    p .+= α
    p /= sum(p)
    return Categorical(p)
end

function get_inner_len_dist(mcmc_output::SisMcmcOutput, α::Float64)
    p = zeros(Int, mcmc_output.model.K_inner)
    for S in mcmc_output.sample
        addcounts!(p, length.(S), 1:mcmc_output.model.K_inner)
    end 
    p /= sum(p)
    p .+= α
    p /= sum(p)
    return Categorical(p)
end 