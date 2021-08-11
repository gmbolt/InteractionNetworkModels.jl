module InteractionNetworkModels

export Path, InteractionSequence, InteractionSequenceSample

# Write your package code here.
# Define some type aliases 
const Path{T} = Vector{T} where {T<:Union{Int, String}}
const InteractionSequence{T} = Vector{Vector{T}} where {T<:Union{Int, String}}
const InteractionSequenceSample{T} = Vector{Vector{Vector{T}}} where {T<:Union{Int, String}}


# Distances
include("Distances/Distances.jl")
include("Distances/InteractionDistances.jl")
include("Distances/InteractionSequenceDistances.jl")

# Types
include("Types/PathDistributions.jl")
include("Types/DataStructures.jl")
include("Types/MetricModels.jl")
include("Types/Posteriors.jl")
include("Types/McmcOutputs.jl")
include("Types/McmcSamplers.jl")
include("Types/distributions.jl")
include("Types/hollywood_model.jl")


# Data Processing
# include("Data Processing/EdgeCounts.jl")
include("Data Processing/VertexCounts.jl")
include("Data Processing/PathSequences.jl")
include("Data Processing/Multigraphs.jl")

# MCMC files
include("MCMC/Model Sampling/multinomial_term_calculation.jl")
include("MCMC/Model Sampling/imcmc_SIS_insert_delete.jl")
include("MCMC/Model Sampling/imcmc_SIM_insert_delete.jl")
# include("MCMC/Model Sampling/imcmc_SIS_insert_delete_fast.jl")
include("MCMC/Model Sampling/imcmc_SPF_edit.jl")
include("MCMC/Model Sampling/summarys.jl")
include("MCMC/Posterior Sampling/iex_interaction_seqs.jl")
include("MCMC/Posterior Sampling/iex_interaction_sets.jl")
include("MCMC/Posterior Sampling/iex_paths.jl")
include("MCMC/Posterior Sampling/parallel_iex_interaction_seqs.jl")


end
