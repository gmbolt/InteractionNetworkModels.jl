using DataFrames, Multisets

export get_vertex_counts

"""
Takes a GroupedDataFrame and a string indicating column name for vertex labels,
outputs a Vector{Multiset}, that is, a multisets of vertex counts for each group
in the data.
"""
function get_vertex_counts(
    X::GroupedDataFrame;
    vertex_col_name::String="page"
    )

    z = Vector{Multiset}()
    for group in X
        push!(z, Multiset(group[:, vertex_col_name]))
    end
    return z
end

#
# get_vertex_counts(df_g)


# ## Using

# function get_vertex_counts(
#     X::GroupedDataFrame;
#     vertex_col_name::String="page"
#     )
#
#     z = Vector{Multiset}()
#     for group in X
#         push!(z, countmap(group[:, vertex_col_name]))
#     end
#     return z
# end

# function hamming(
#     X::Dict{T,Int} where T,
#     Y::Dict{T,Int} where T
#     )
#     z = 0.0
#     for k in intersect(keys(X), keys(Y))
#         z += abs(X[k]-Y[k])
#     end
#     for k in setdiff(keys(X), keys(Y))
#         z += X[k]
#     end
#     for k in setdiff(keys(Y), keys(X))
#         z += Y[k]
#     end
#     return z
# end
#
#
# function sum_values(d::Dict{T, Int} where T)
#     z = 0
#     for val in values(d)
#         z += val
#     end
#     return z
# end
#
# sum_values(tmp1)
#
# # Jaccard distance, this was cheaper than using steinhaus on hamming
# function jaccard(
#     X::Dict{T, Int} where T,
#     Y::Dict{T, Int} where T
#     )
#
#     a = 0.0
#     for k in intersect(keys(X), keys(Y))
#         a += min(X[k], Y[k])
#     end
#     b = sum_values(mergewith(max, X, Y))
#     return 1-a/b
# end
#
#
# d1 = Dict("a"=>10, "b"=>2)
# d2 = Dict("a"=>1, "d"=>20)
#
# using BenchmarkTools
#
# @time hamming(tmp1, tmp2)
#
# @btime jaccard(tmp1, tmp2)
#
# ## Distance Matrices
#
#
# @time
#
# tmp = get_dist_mat(data)
#
# using UMAP, DataFrames, StatsPlots
#
# Random.seed!(1234);  # Seed for repoduceability
#
# min_dist = 0.001
# n_neighbors = 50
# embedding = umap(tmp;
#     metric=:precomputed,
#     min_dist=min_dist,
#     n_neighbors=n_neighbors)
#
# embedding = DataFrame(embedding')  # Make dataframe
#
# @df embedding scatter(
#     :x1, :x2,
#     ms=2.3, markerstrokewidth=0, markeralpha=0.4,
#     leg=false,
#     size = (800,600))
#
#
# multiset(
