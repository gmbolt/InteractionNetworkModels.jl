export SpfMcmcSampler, SisMcmcSampler, SimMcmcSampler
export SpfInvolutiveMcmcCentSubseq, SpfInvolutiveMcmcEdit
# ====================
#   SPF 
# ====================

# From Model
abstract type SpfMcmcSampler end

struct SpfInvolutiveMcmcEdit <: SpfMcmcSampler
    ν::Int 
    desired_samples::Int
    burn_in::Int 
    lag::Int
    par_info::Dict
    curr::Vector{Int} # Storage for current val
    prop::Vector{Int} # Storage for proposed val
    ind_del::Vector{Int} # Storage for indexing of deletions
    ind_add::Vector{Int} # Storage for indexing of additions
    vals::Vector{Int} # Storgae for new values to insert
    function SpfInvolutiveMcmcEdit(
        ;ν=4, desired_samples=1000, burn_in=0, lag=1
    )
    # req_samples = burn_in + 1 + (desired_samples - 1) * lag
    curr = Int[]
    prop = Int[]
    ind_del = zeros(Int, ν)
    ind_add = zeros(Int, ν)
    vals = zeros(Int, ν)
    par_info = Dict()
    par_info[:ν] = "(maximum number of edit operations)"
    new(
        ν, 
        desired_samples, burn_in, lag, par_info, 
        curr, prop, ind_del, ind_add, vals
        )
    end 
end 

function Base.show(io::IO, sampler::SpfInvolutiveMcmcEdit)
    title = "MCMC Sampler for Spherical Path Family (SPF) via Edit Operations"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 1
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 


struct SpfInvolutiveMcmcCentSubseq <: SpfMcmcSampler
    p::Real
    ν::Int
    desired_samples::Int
    burn_in::Int 
    lag::Int
    par_info::Dict
    function SpfInvolutiveMcmcCentSubseq(
        ;p=0.6, ν=4, desired_samples=1000, burn_in=0, lag=1
    )
    par_info = Dict()
    par_info[:p] = "(preserved subseq size)"
    par_info[:ν] = "(diff in path length)"
    new(p, ν, desired_samples, burn_in, lag, par_info)
    end 
end 

function Base.show(io::IO, sampler::SpfInvolutiveMcmcCentSubseq)
    title = "MCMC Sampler for Spherical Path Family (SPF) via Subsequence Preservation"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    num_of_pars = 2
    for par in fieldnames(typeof(sampler))[1:num_of_pars]
        println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
    end 
    println(io, "\nDefault output parameters:")
    for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+3)]
        println(io, par, " = $(getfield(sampler, par))  ")
    end 
end 

# From Posterior 

# struct SpfPosteriorInvExMcmc
#     mcmc_sampler::SpfMcmcSampler
#     ν::Int
#     vdist::Vector{Float64} # Probability Vector for vertex proposal
#     α::Float64
#     desired_samples::Int
#     burn_in::Int 
#     lag::Int
#     par_info::Dict
#     curr::Vector{Int} # Storage for current val
#     prop::Vector{Int} # Storage for proposed val
#     ind_del::Vector{Int} # Storage for indexing of deletions
#     ind_add::Vector{Int} # Storage for indexing of additions
#     vals::Vector{Int} # Storage for new values to insert
#     function SpfPosteriorInvExMcmc(
#         mcmc_sampler::SpfMcmcSampler;
#         ν::Int=2,
#         vdist::Vector{Float64}=Float64[], 
#         α::Float64=0.0,
#         desired_samples::Int=100, 
#         burn_in::Int=0, 
#         lag::Int=1
#     )
#     curr = Int[]
#     prop = Int[]
#     ind_del = zeros(Int, ν)
#     ind_add = zeros(Int, ν)
#     vals = zeros(Int, ν)
#     par_info = Dict()
#     par_info[:mcmc_sampler] = "(mcmc sampler for auxiliary data)"
#     par_info[:ν] = "(maximum number of edit operations)"
#     par_info[:vdist] = "(proposal distribution for vertices, a probability vector)"
#     par_info[:α] = "(informed proposal damping - higher value => more uniform/less informed)"
#     new(
#         mcmc_sampler, ν, vdist, α,
#         desired_samples, burn_in, lag, 
#         par_info,
#         curr, prop, ind_del, ind_add, vals
#         )
#     end 
# end 

# function Base.show(io::IO, sampler::SpfPosteriorInvExMcmc)
#     title = "MCMC Sampler for SPF Posterior via iExchange / Double iMCMC Algorithm"
#     n = length(title)
#     println(io, title)
#     println(io, "-"^n)
#     println(io, "Parameters:")
#     num_of_pars = 4
#     for par in fieldnames(typeof(sampler))[1:num_of_pars]
#         println(io, par, " = $(getfield(sampler, par))  ", sampler.par_info[par])
#     end 
#     num_of_output_pars = 3
#     println(io, "\nDefault output parameters:")
#     for par in fieldnames(typeof(sampler))[(num_of_pars+1):(num_of_pars+num_of_output_pars)]
#         println(io, par, " = $(getfield(sampler, par))  ")
#     end 
# end 

# function inform_propsal!(
#     sampler::SpfPosteriorInvExMcmc, 
#     data::Vector{Path{Int}}
#     )

#     V = maximum(vertices(data))
#     μ = vertex_counts(data, 1:V) 
#     μ /= sum(μ)
#     resize!(sample.vdist, V)
#     for i in 1:V
#         sampler.vdist[i] = μ[i]
#     end 
# end 




