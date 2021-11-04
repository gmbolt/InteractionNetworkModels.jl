module InteractionNetworkModels

using Distributed

# Write your package code here.
# Alises for referring to Paths/Interaction Sequences/Samples of Interaction Sequences
# (purely for code readability)
include("aliases.jl")

# Distances
include("Distances/helpers.jl")
include("Distances/interactions.jl")
include("Distances/interaction_sequences.jl")
include("Distances/interaction_multisets.jl")
include("Distances/graph_distances.jl")

# Types
include("Types/PathDistributions.jl")
include("Types/DataStructures.jl")
include("Types/MetricModels.jl")
include("Types/distributions.jl")
include("Types/hollywood_model.jl")


# Data Processing
# include("Data Processing/EdgeCounts.jl")
include("Data Processing/VertexCounts.jl")
include("Data Processing/PathSequences.jl")
include("Data Processing/Multigraphs.jl")

# MCMC files
include("MCMC/Model Sampling/types_mcmc_model_samplers.jl")
include("MCMC/Model Sampling/types_mcmc_model_outputs.jl")
include("MCMC/Model Sampling/multinomial_term_calculation.jl")
include("MCMC/Model Sampling/imcmc_SIS.jl")
include("MCMC/Model Sampling/imcmc_SIS_gibbs.jl")
include("MCMC/Model Sampling/imcmc_SIM.jl")
include("MCMC/Model Sampling/imcmc_SIM_gibbs.jl")
# include("MCMC/Model Sampling/imcmc_SIS_insert_delete_fast.jl")
include("MCMC/Model Sampling/imcmc_SPF.jl")
include("MCMC/Model Sampling/summarys.jl")
include("MCMC/Posterior Sampling/types_posteriors.jl")
include("MCMC/Posterior Sampling/types_mcmc_posterior_samplers.jl")
include("MCMC/Posterior Sampling/types_mcmc_posterior_outputs.jl")
include("MCMC/Posterior Sampling/cooccurrence_matrices.jl")
include("MCMC/Posterior Sampling/iex_paths.jl")
include("MCMC/Posterior Sampling/iex_SIS.jl")
include("MCMC/Posterior Sampling/iex_SIM.jl")
include("MCMC/Posterior Sampling/summarys.jl")
include("MCMC/Posterior Sampling/missing_entry_predictives.jl")
# include("MCMC/Posterior Sampling/parallel_iex_interaction_seqs.jl")
include("MCMC/Posterior Sampling/parallel_helpers.jl")


end
