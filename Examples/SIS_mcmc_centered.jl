using InteractionNetworkModels, BenchmarkTools, Distributions
using Plots, StatsBase

mcmc = SisMcmcSplitMerge(
    β=0.3, p=0.7, η=0.5,
    ν_td=1
)


model_mode = Hollywood(-3.0, Poisson(7), 10)
S = [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
V = 1:20
# d = FastEditDistance(FastLCS(51),51)
d = DTW(FastLCS(100))
model = SIS(S, 3.9, d, V, 50, 50)

S2 = [[1,1,1,1], [2,2,3,11,2,1,2], [2,2,2,2], [3,3,3,3]]
d(S,S2)

x = mcmc(model, desired_samples=100)
plot(x)
model.mode


S_prop = deepcopy(S)
inds = [2]
multiple_adj_split_noisy!(
    S_prop, 
    [Int[] for i in 1:100],
    inds, 
    Geometric(0.7), 
    0.7, 1:10, 
    1,10
)
S_prop

# Base dimension range performance
# --------------------------------
xl,xu  = (1,100)
r = DimensionRange(xl,xu)
y=10
@btime notin(r,y)

function test(x::Real, l::Real, u::Real)
    return (x >= l) & (x<=u)
end 
@btime test(y, xl, xu)

# But do this and we get performace back
rxl = r.l
rxu = r.u
@btime test(y, rxl, rxu)

# Undo a merge 

function test_undo_merge!(
    x::Path{Int},
    ind::Vector{Int}
    )
    live_index = 0
    for i in ind   
        x[i+live_index] = 0
        insert!(x, i+live_index+1, 0)
        live_index += 1
    end 
end 

x = [1,2,3,4,5]
test_undo_merge!(x, [1,5])
x

function test_undo_split!(
    x::Path{Int}, 
    ind::Vector{Int}
    )   

    for i in ind 
        tmp = popat!(x, i+1)
        x[i] = 0
    end 

end

x = [1,1,2,3,4,5,5]
ind=[1,5]

test_undo_split!(x, [1,5])
x

function get_cond_dist(
    S::InteractionSequence{Int};
    α::Float64=0.0,
    V::Int=length(unique(vcat(S...)))
    )
    tot_len = 0
    c = fill(α, V)
    for path in S 
        for v in path 
            c[v] += 1.0 
            tot_len += 1
        end 
    end 
    c ./= (tot_len + α * V)
    return c
end


get_cond_dist(S, V=10)

S = [[1,2,1,2,1], [2,2,2,2], [3,3,3,3,10]]

@time get_cond_dist(S, V=10, α=10.0)

d = Multinomial(10, [0.9,0.1])

@time rand(Multinomial(1, [0.9,0.1]))

@time rand(Categorical([0.1,0.9]))