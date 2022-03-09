using InteractionNetworkModels, Distributions, BenchmarkTools
using StatsBase, Plots, StatsPlots

# Testing truncated geom difference 
p = 0.7
n = 30
d = TrGeometric(p,n)
a = Geometric(p)
data = [n-rand(d)+rand(a) for i in 1:1000] 
bar(counts(data, 0:2*n))

p = 0.8
p_del = TrGeometric(p,0,length(curr))
p_ins = Geometric(p)
η = 0.3
V = 1:10

ind_del = [1,2]
ind_add_1 = [1]
ind_add_2 = [1,6]
vals_1 = [10]
vals_2 = [9,9]
curr = [1,1,1,1,1,1]
prop1 = copy(curr)
prop2 = copy(curr)

# In place (with two proposals given)
@time split_noisy!(
    curr, 
    prop1, prop2,
    ind_del, 
    ind_add_1, ind_add_2, 
    vals_1, vals_2, 
    η, V
)
curr,prop1,prop2

# When given current value and storage
curr = [1,2,1,2,1,2]
prop1 = copy(curr) # Current val 
prop2 = Int[] # Storage for new val 
@time split_noisy!(
    prop1, 
    prop2,
    ind_del, 
    ind_add_1, ind_add_2, 
    vals_1, vals_2, 
    η, V
)
prop1, prop2

# Standard 
split_noisy(
    curr, 
    ind_del, 
    ind_add_1, ind_add_2, 
    vals_1, vals_2, 
    η, V
)

# With auxiliary sampling
curr = [1,2,1,2,1,2]
p = 0.7
p_del = TrGeometric(p,0,length(curr))
p_ins = Geometric(p)
η = 0.7
V = 1:10
prop1, prop2 = (copy(curr), copy(curr))
# In place
rand_split_noisy!(
    curr, 
    prop1, prop2, 
    p_del, p_ins, 
    η, V
)
curr, prop1, prop2

# With alloc 
rand_split_noisy(
    curr, 
    p_del, p_ins, 
    η, V
)

# ========
# Merging 
# =======

# With keep indexing 
curr1 = [1,1,1,1]
curr2 = [2,2,2,2]
ind1_keep = [1,2,3]
ind2_keep = [1,3,4]
ind_add = [1,4]
vals = [10,10]
prop = zeros(Int, length(ind1_keep))
@time merge_noisy!(
    curr1, curr2, 
    prop,
    ind1_keep,ind2_keep, 
    ind_add, vals, 
)
curr1, curr2, prop

@time merge_noisy(
    curr1, curr2, 
    ind1_keep, ind2_keep, 
    ind_add, vals
)

# With deletion indexing 
ind1_del = [1,4]
ind2_del = [1]
ind_add = [1,4]
vals = [10,10]
curr1 = [2,1,2,1]
curr2 = [2,2,2]
@time merge_noisy_ins_dels!(
    curr1, curr2, 
    ind1_del, ind2_del, 
    ind_add, vals
)
curr1, curr2


# Random 
p = 0.8
V = 1:10
η = 0.9
curr1 = [1,1,1]
curr2 = [2,2,2,2]
p_ins = Geometric(p)
p_del = TrGeometric(p,0,min(length(curr1), length(curr2)))
@time rand_merge_noisy_ins_dels!(
    curr1, curr2, 
    p_del, p_ins, 
    V, 
    η
)
curr1, curr2

# =====================
# Multiple split/merges
# =====================

# Multiple splits 
ind_split = [1,3]
p = 0.5
p_ins = Geometric(p)
η = 0.5
V = 1:10
S_curr = [[1,1,1], [2,2,2], [3,3,3]]
S_store = [Int[] for i in 1:4]
@time multiple_adj_split_noisy!(
    S_curr, 
    S_store, 
    ind_split, 
    p_ins, η, V
)
S_curr
S_store


# Multiple merges 
ind_merge = [1,3]
p = 0.9
p_ins = Geometric(p)
η = 0.1
V = 1:10
S_curr = [[1,1,1], [1,1,2], [3,3,3], [2,3,2], [3,3,3]]
S_store = [Int[] for i in 1:2]
@time multiple_adj_merge_noisy!(
    S_curr, 
    S_store, 
    ind_merge, 
    p_ins, η, V
)
S_curr
S_store

# With loc sampling 
# ------------------

# Split
p = 0.4
p_ins = Geometric(p)
η = 0.7
V = 1:10
ν_td = 10
S_curr = [[1,1,1], [2,2,2], [3,3,3]]
S_store = [Int[] for i in 1:4]
@time rand_multiple_adj_split_noisy!(
    S_curr, 
    S_store, 
    ν_td, 
    p_ins, η, V
)
S_curr
S_store

# Merge
p = 0.99
p_ins = Geometric(p)
η = 0.1
V = 1:10
ν_td = 10
S_curr = [[1,1,1], [2,2,2], [3,3,3]]
S_store = [Int[] for i in 1:4]
@time rand_multiple_adj_merge_noisy!(
    S_curr, 
    S_store, 
    ν_td, 
    p_ins, η, V
)
S_curr
S_store

# Subsequence sampling 
n = 10
k = 5
ind = zeros(Int,k)
@btime StatsBase.seqsample_a!(1:n, ind)
@btime StatsBase.seqsample_c!(1:n, ind)
@btime StatsBase.seqsample_d!(1:n, ind)

StatsBase.seqsample_a!(1:n, ind)
ind
StatsBase.seqsample_c!(1:n, ind)
ind 
StatsBase.seqsample_d!(1:n, ind)
ind

