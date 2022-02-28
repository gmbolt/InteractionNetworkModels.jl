using Multisets, IterTools

function print_map_est(output::SisPosteriorMcmcOutput; top_num::Int=5) 
    d = proportionmap(output.S_sample)
    # for key in keys(d)
    #     d[key] /= length(output.S_sample)
    # end 
    props = sort(collect(d), by=x->x[2], rev=true)
    title = "\nPosterior probability of mode"
    println(title)
    println("-"^length(title), "\n")
    for i in 1:min(top_num, length(d))
        println(props[i][2],"  ", props[i][1])
    end    
    println("\n...showing top $(min(top_num, length(d))) interaction sequences.")
end 

function print_map_est(sample::InteractionSequenceSample; top_num::Int=5) 
    d = proportionmap(sample)
    # for key in keys(d)
    #     d[key] /= length(output.S_sample)
    # end 
    props = sort(collect(d), by=x->x[2], rev=true)
    title = "\nPosterior probability of mode"
    println(title)
    println("-"^length(title), "\n")
    for i in 1:min(top_num, length(d))
        println(props[i][2],"  ", props[i][1])
    end    
    println("\n...showing top $(min(top_num, length(d))) interaction sequences.")
end 

function mean_dists_summary(
    samples::Vector{InteractionSequenceSample{T}},
    d::Union{Metric,Metric}
    ) where {T<:Union{Int,String}}

    tot_var = sample_frechet_var(
            vcat(samples...), 
            d, 
            with_memory=true, show_progress=true
        )
    mean_of_var = mean(
        x->sample_frechet_var(
            x, 
            d, 
            with_memory=true, show_progress=true), samples
    )
    means = map(x->sample_frechet_mean(x, d, with_memory=true)[1], samples)
    var_of_mean = sample_frechet_var(means, d)

    out = Dict{String,Float64}(
        "tot_var" => tot_var,
        "mean_of_vars" => mean_of_var,
        "var_of_means" => var_of_mean
    )
    return out
end 

function mean_dists_summary(
    chains::Vector{SisPosteriorMcmcOutput}
    )
    samples = [x.S_sample for x in chains]
    d = chains[1].posterior.dist
    return mean_dists_summary(samples, d)
end 


function print_map_est(output::Union{SimPosteriorMcmcOutput,SimPosteriorModeConditionalMcmcOutput}; top_num::Int=5) 
    multiset_sample = Multiset.(output.S_sample)
    d = proportionmap(multiset_sample)
    # for key in keys(d)
    #     d[key] /= length(output.S_sample)
    # end 
    props = sort(collect(d), by=x->x[2], rev=true)
    title = "\nPosterior probability of mode"
    println(title)
    println("-"^length(title), "\n")
    for i in 1:min(top_num, length(d))
        println(props[i][2],"  ", props[i][1])
    end    
    println("\n...showing top $(min(top_num, length(d))) interaction multisets.")
end 
