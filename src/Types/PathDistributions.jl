using Distributions

export PathDistribution, PathPseudoUniform, PathCooccurrence

abstract type PathDistribution{T<:Union{Int,String}} end 

struct PathPseudoUniform{T<:Union{Int, String}} <: PathDistribution{T}
    vertex_set::Vector{T}
    length_dist::DiscreteUnivariateDistribution
end 

struct PathCooccurrence{T<:Union{Int,String}} <: PathDistribution{T}
    vertex_set::Vector{T}
    length_dist::DiscreteUnivariateDistribution
    initial_dist::DiscreteUnivariateDistribution
    P::Matrix{Float64}
    P_dists::Dict{Int,Categorical}
    function PathCooccurrence(
        vertex_set::Vector{S},
        length_dist::DiscreteUnivariateDistribution,
        initial_dist::DiscreteUnivariateDistribution, 
        P::Matrix{Float64}
        ) where {S<:Union{Int,String}}
        @assert length(vertex_set) == size(P)[1] "Dimension missmatch. Check vertex set and transition matrix dimensions agree."
        dists = Dict{Int, Categorical}()
        for i in 1:length(vertex_set)
            dists[i] = Categorical(P[:,i])
        end 
        new{S}(vertex_set, length_dist, initial_dist, P, dists)
    end 
end 

# When no vertex set it given we assume integer vertices
function PathCooccurrence(
    length_dist::DiscreteUnivariateDistribution, 
    initial_dist::DiscreteUnivariateDistribution, 
    P::Matrix{Float64}
    ) 
    return PathCooccurrence(collect(1:size(P)[1]), length_dist, initial_dist, P)
end 


function Base.rand(d::PathPseudoUniform)
    m = rand(d.length_dist)
    return rand(d.vertex_set, m)
end 

function rand!(p::Vector{T}, d::PathPseudoUniform{T}) where {T<:Union{Int,String}}

    m = rand(d.length_dist)
    resize!(p, m)
    sample!(d.vertex_set, p)

end 

function Base.rand(d::PathCooccurrence{String})
    m = rand(d.length_dist) # Sample length 
    log_prob = logpdf(d.length_dist, m) # Intiialise log probability
    unique_vals = Set{Int}() # To store unique vertices observed in path
    z = Vector{String}(undef, m) # Initialisation of storage
    
    tmp = rand(d.initial_dist) # Sample initial vertex index
    z[1] = d.vertex_set[tmp] # Index the corresponding vertex 
    log_prob += logpdf(d.initial_dist, tmp) # Increment log probability
    push!(unique_vals, tmp) # Store value sampled

    for i in 2:m
        component = rand(unique_vals) # Sample component
        tmp = rand(d.P_dists[component]) # Sample vertex index
        z[i] = d.vertex_set[tmp] # Store vertex value
        # Increment log probability (sum over unique values and normalise by number of unique values)
        for val in unique_vals
            log_prob += d.P[val,tmp]/length(unique_vals)
        end 
        push!(unique_vals, tmp) # Update unqiue values seen 
    end 
    return z, log_prob
end 

function Base.rand(d::PathCooccurrence{Int})
    m = rand(d.length_dist) # Sample length
    log_prob = logpdf(d.length_dist, m)  # Inistialise log probability

    unique_vals = Set{Int}() # Storage unique vertices observed in path
    z = Path(Vector{Int}(undef, m)) # Storage for sampled observation

    # Draw initial value
    tmp = rand(d.initial_dist) # Sample value
    z[1] = tmp # Store it 
    log_prob += logpdf(d.initial_dist, tmp) # Increment log probability
    push!(unique_vals, tmp) # Store value sampled

    for i in 2:m
        component = rand(unique_vals)  # Sample component (which row/col of coocurrence matrix)
        tmp = rand(d.P_dists[component]) # Sample value from component distribution
        z[i] = tmp # Store value
        # Increment log probability (sum over unique values)
        for val in unique_vals
            log_prob += d.P[val,tmp]/length(unique_vals)
        end 
        push!(unique_vals, tmp) # Store value sampled
    end 
    return z, log_prob
end 



function Distributions.logpdf(d::PathPseudoUniform{T}, p::Path{T}) where {T<:Union{Int,String}}
    m = length(p)
    return logpdf(d.length_dist, m) - m*log(length(d.vertex_set))
end 


