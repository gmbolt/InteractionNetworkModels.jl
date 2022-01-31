using Distributions
export MarkovChain, DirichletRows
export draw_sample, draw_sample!
export get_transition_matrix

struct MarkovChain 
    P::Matrix{Float64}
    P_c::Matrix{Float64}
    mu::Categorical
end 
MarkovChain(P::Matrix{Float64}, x::Int) = MarkovChain(
    P, 
    cumsum(P, dims=2), 
    Categorical([i==x ? 1.0 : 0.0 for i in 1:size(P,1)])
)

MarkovChain(P::Matrix{Float64}, μ::Categorical) = MarkovChain(
    P, 
    cumsum(P, dims=2), 
    μ
)

MarkovChain(P::Matrix{Float64}) = MarkovChain(
    P, 
    cumsum(P, dims=2), 
    Categorical([1/size(P,1) for i in 1:size(P,1)])
)

function rand_categorical(μ_cusum::AbstractArray{Float64})
    β = rand()
    for i in 1:length(μ_cusum)
        if β < μ_cusum[i]
            if i == 1
                return i
            else 
                return i
            end 
        else 
            continue 
        end 
    end 
end 

function rand_categorical_with_prob(μ_cusum::AbstractArray{Float64})
    β = rand()
    for i in 1:length(μ_cusum)
        if β < μ_cusum[i]
            if i == 1
                return i, μ_cusum[i]
            else 
                return i, μ_cusum[i] - μ_cusum[i-1]
            end 
        else 
            continue 
        end 
    end 
end 

function draw_sample!(
    out::Vector{Int},
    model::MarkovChain, 
    n::Int)

    # Sample initial value 
    curr = rand(model.mu)
    out[1] = curr
    for i in 2:n
        @views curr = rand_categorical(model.P_c[curr,:])
        out[i] = curr
    end 
end 

function draw_sample(
    model::MarkovChain, 
    n::Int)

    out = zeros(Int, n)
    draw_sample!(out, model, n)
    return out
end 

struct DirichletRows
    A::Matrix{Float64}
end 

function draw_sample(model::DirichletRows)
    P = [rand(Gamma(a)) for a in model.A]
    P ./= sum(P, dims=2)
    return P
end 

function draw_sample!(
    out::Vector{Matrix{Float64}},
    model::DirichletRows
    )

    for P in out 
        for j in 1:size(P,2)
            for i in 1:size(P,1)
                P[i,j] = rand(Gamma(model.A[i,j]))
            end 
        end 
        P ./= sum(P, dims=2)
    end 
end 

function draw_sample(
    model::DirichletRows, 
    n::Int
    )

    out = [similar(model.A) for i in 1:n]
    draw_sample!(out,model)
    return out 
end 


function get_transition_matrix(
    data::Vector{Int},
    V::Int
    )

    T = zeros(Int, V, V)
    @inbounds for i in Iterators.rest(eachindex(data),1)
        T[data[i-1], data[i]] += 1
    end 
    return T
end 

function get_transition_matrix(
    data::Vector{Int}
    )

    V = maximum(data)
    return get_transition_matrix(data, V)
end 
